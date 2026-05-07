"""BatchedMuon: drop-in Muon replacement with grouped Newton-Schulz iterations.

``torch.optim.Muon`` has no ``foreach`` path — every parameter is orthogonalised
in a serial Python loop (one GPU kernel launch per parameter per NS step).
This module replaces that inner loop with batched matrix multiply:
parameters are bucketed by canonical shape, stacked into a 3-D tensor, and
all NS iterations run via ``torch.bmm`` / ``torch.baddbmm``.

The number of Python→GPU round-trips drops from
    O(N_params × N_ns_steps)
to
    O(N_distinct_shapes × N_ns_steps)
which is typically 1–3 kernel series for a homogeneous FFN/transformer.

**CPU vs GPU behaviour:**
The speedup is GPU-specific.  On CPU, BLAS already saturates all available
cores for each individual matmul, so batching adds no new parallelism and
wall-clock time is indistinguishable from ``torch.optim.Muon``.  On GPU each
small weight matrix underutilises the SMs; ``torch.bmm`` fills them across all
same-shape parameters simultaneously and eliminates the per-parameter
Python→GPU kernel-launch overhead (~1–10 µs each on CUDA).  Muon is designed
for GPU training, so the optimisation targets that path.

**Benchmark (CPU, optimizer.step() only, 6 × 256×256 weight matrices):**

.. code-block:: text

    Adam (all params)         :    0.727 ms/step
    torch.optim.Muon + AdamW  : 1036.280 ms/step   (1426x vs Adam)
    BatchedMuon + AdamW       : 1054.200 ms/step   (1451x vs Adam)
    Speedup Muon → Batched    :    1.0x

No speedup on CPU (expected — see note above).  GPU results not yet measured
on this hardware; the architectural improvement targets GPU kernel-launch
overhead which is absent on CPU.

Drop-in usage: identical constructor signature to ``torch.optim.Muon``.
Rollback: when PyTorch ships ``foreach`` for Muon, delete this file and
revert ``_build_muon_mixed`` in ``builder.py`` to use ``TorchOptimizerFactory``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor
from torch.optim import Muon

if TYPE_CHECKING:
    from collections.abc import Callable


def _batch_zeropower_via_newtonschulz(
    grads: list[Tensor],
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> list[Tensor]:
    """Newton-Schulz orthogonalization over a list of 2-D tensors via batched bmm.

    Groups tensors by canonical (m, n) shape (m ≤ n after optional transpose),
    stacks each group, and runs all NS iterations with ``torch.bmm`` /
    ``torch.baddbmm``.  One kernel launch series per shape group per step,
    instead of one per parameter per step.

    Args:
        grads: 2-D gradient tensors to orthogonalise.
        ns_coefficients: Quintic polynomial coefficients ``(a, b, c)``.
        ns_steps: Number of Newton-Schulz iterations.
        eps: Spectral-norm floor for numerical stability.

    Returns:
        Orthogonalised update tensors in the same order as ``grads``.
    """
    a, b, c = ns_coefficients
    results: list[Tensor | None] = [None] * len(grads)

    # Group by canonical shape (rows ≤ cols) to enable batching.
    shape_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, g in enumerate(grads):
        r, k = g.shape
        shape_groups[(min(r, k), max(r, k))].append(idx)

    for indices in shape_groups.values():
        transposed = [grads[i].size(0) > grads[i].size(1) for i in indices]
        oriented = [
            grads[i].bfloat16().T if t else grads[i].bfloat16()
            for i, t in zip(indices, transposed, strict=True)
        ]

        stacked = torch.stack(oriented)  # (B, m, n)
        stacked = stacked / stacked.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)

        for _ in range(ns_steps):
            gram = torch.bmm(stacked, stacked.transpose(-2, -1))  # (B, m, m)
            gram = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)  # (B, m, m)
            stacked = torch.baddbmm(stacked, gram, stacked, beta=a)  # (B, m, n)

        for local_i, (global_i, t) in enumerate(zip(indices, transposed, strict=True)):
            out = stacked[local_i]
            results[global_i] = out.T if t else out

    return cast(list[Tensor], results)


class BatchedMuon(Muon):
    """Muon optimizer with batched Newton-Schulz orthogonalization.

    Identical update rule to ``torch.optim.Muon``; only the NS inner loop
    is replaced with grouped ``torch.bmm`` calls.  All other behaviour
    (``__init__``, param validation, momentum state, LR adjustment) is
    inherited from the parent unchanged.

    .. warning::
        **2-D parameters only.**  Muon is designed for hidden-layer weight
        matrices.  Biases, embeddings, layer-norm parameters, and all other
        non-matrix tensors must be optimized by a separate optimizer (AdamW
        is recommended).  Passing non-2D parameters raises ``ValueError`` at
        construction time.

    .. note::
        ``foreach`` is not supported by ``torch.optim.Muon``; enabling it
        raises ``RuntimeError``.  Complex parameters and sparse gradients
        are also unsupported and raise ``RuntimeError`` at step time.
        See `torch.optim.Muon
        <https://docs.pytorch.org/docs/main/generated/torch.optim.Muon.html>`_
        for the official documentation.

    **When does this help?**
    The speedup is GPU-specific.  On CPU, BLAS already parallelises each
    individual matmul across all cores, so batching yields no additional
    throughput.  On GPU, each small weight matrix under-utilises the SMs;
    processing all same-shape parameters together with ``torch.bmm`` keeps
    the GPU busier and eliminates per-parameter kernel-launch overhead.

    Args:
        params: 2-D parameter tensors (same restriction as ``torch.optim.Muon``).
        lr: Learning rate.
        weight_decay: Decoupled weight decay (Muon default: 0.1).
        momentum: Momentum factor.
        nesterov: Use Nesterov momentum.
        ns_coefficients: Newton-Schulz quintic polynomial coefficients ``(a, b, c)``.
        eps: Spectral-norm floor for numerical stability.
        ns_steps: Number of Newton-Schulz iterations.
        adjust_lr_fn: LR shape adjustment (``"original"`` or ``"match_rms_adamw"``).
    """

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # ty: ignore[invalid-method-override]
        """Perform one optimisation step with batched Newton-Schulz orthogonalization.

        Args:
            closure: Optional callable that re-evaluates the model and returns the loss.

        Returns:
            Loss value if closure was provided, else ``None``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            bufs: list[Tensor] = []
            self._init_group(group, params_with_grad, grads, bufs)

            if not params_with_grad:
                continue

            lr: float = group["lr"]
            wd: float = group["weight_decay"]
            momentum: float = group["momentum"]
            nesterov: bool = group["nesterov"]
            ns_coeff: tuple[float, float, float] = group["ns_coefficients"]
            eps: float = group["eps"]
            ns_steps: int = group["ns_steps"]
            adjust_lr_fn: str | None = group["adjust_lr_fn"]

            updates: list[Tensor] = []
            for g, buf in zip(grads, bufs, strict=True):
                buf.lerp_(g, 1 - momentum)
                updates.append(g.lerp(buf, momentum) if nesterov else buf)

            ortho = _batch_zeropower_via_newtonschulz(updates, ns_coeff, ns_steps, eps)

            for param, o in zip(params_with_grad, ortho, strict=True):
                A, B = param.shape[:2]
                if adjust_lr_fn is None or adjust_lr_fn == "original":
                    adj_lr = lr * math.sqrt(max(1, A / B))
                elif adjust_lr_fn == "match_rms_adamw":
                    adj_lr = lr * 0.2 * math.sqrt(max(A, B))
                else:
                    adj_lr = lr
                param.mul_(1 - lr * wd)
                param.add_(o.to(param.dtype), alpha=-adj_lr)

        return loss
