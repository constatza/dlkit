"""Array/dense postprocessing utilities.

Functions in this module focus on dense outputs (tensors, ndarrays) and
simple dict-of-tensors patterns that are common in batch predictions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .core import NotStackableError


def to_numpy(obj: Any) -> Any:
    """Recursively convert tensors to numpy arrays.

    - torch.Tensor → detached cpu numpy
    - dict/list/tuple → recursive conversion
    - numpy/scalars → unchanged
    """
    try:
        import torch  # lazy import

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
    except Exception:
        pass

    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, dict):
        return {k: to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_numpy(v) for v in obj]
    return obj


def _stack_list_tensors(items: Sequence[Any], *, mode: str) -> np.ndarray | list[np.ndarray]:
    """Stack a list of tensors/arrays according to mode.

    mode="stack": strict concatenate along dim=0; raise if incompatible
    mode="list": return list of arrays
    mode="pad": pad along dim=0 to max length with zeros
    """
    try:
        import torch

        tensors: list[torch.Tensor] = []
        for x in items:
            if isinstance(x, torch.Tensor):
                tensors.append(x)
            elif isinstance(x, np.ndarray):
                tensors.append(torch.from_numpy(x))
            else:
                raise NotStackableError(f"Unsupported type {type(x).__name__} for tensor stacking")

        if mode == "list":
            return [t.detach().cpu().numpy() for t in tensors]

        if mode == "stack":
            # validate shapes except dim 0
            base_shape = tensors[0].shape[1:]
            for t in tensors[1:]:
                if t.shape[1:] != base_shape:
                    raise NotStackableError(
                        f"Cannot stack: shape mismatch {t.shape} vs (*, {base_shape})"
                    )
            return torch.cat(tensors, dim=0).detach().cpu().numpy()

        if mode == "pad":
            max_len = max(int(t.shape[0]) for t in tensors)
            padded = []
            for t in tensors:
                if int(t.shape[0]) == max_len:
                    padded.append(t)
                else:
                    pad_len = max_len - int(t.shape[0])
                    pad_shape = (pad_len, *t.shape[1:])
                    pad = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
                    padded.append(torch.cat([t, pad], dim=0))
            return torch.stack(padded, dim=0).detach().cpu().numpy()

        raise ValueError(f"Unknown mode: {mode}")
    except NotStackableError:
        raise
    except Exception as e:  # pragma: no cover - defensive
        raise NotStackableError(str(e)) from e


def stack_batches(
    batches: Sequence[Any], *, keys: Sequence[str] | None = None, mode: str = "list"
) -> Any:
    """Stack a sequence of batches (dense) into arrays.

    Supported inputs:
    - list[Tensor] or list[np.ndarray]
    - list[dict[str, Tensor|np.ndarray]]
    - list[scalars]

    modes:
    - "list" (default): return a list of numpy arrays
    - "stack": strict concatenate along batch axis (raises on mismatch)
    - "pad": pad to max length along batch axis and stack with an outer batch
    """
    if not batches:
        return []

    first = batches[0]

    # List of tensors/arrays
    try:
        import torch

        if isinstance(first, (torch.Tensor, np.ndarray)):
            return _stack_list_tensors(batches, mode=mode)
    except Exception:
        if isinstance(first, np.ndarray):
            return _stack_list_tensors(batches, mode=mode)

    # List of dicts
    if isinstance(first, Mapping):
        all_keys = set().union(*(b.keys() for b in batches))
        if keys is not None:
            use_keys = [k for k in keys if k in all_keys]
        else:
            use_keys = sorted(all_keys)

        out: dict[str, Any] = {}
        for k in use_keys:
            values = [b[k] for b in batches if k in b]
            # If values are tensor-like → stack according to mode
            try:
                out[k] = _stack_list_tensors(values, mode=mode)
            except NotStackableError:
                out[k] = [to_numpy(v) for v in values]
        return out

    # Scalars → numpy array
    return np.asarray(batches)


def to_plot_data(
    preds: Sequence[Any] | Mapping[str, Any] | Any,
    targets: Sequence[Any] | Mapping[str, Any] | None = None,
) -> Any:
    """Convert predictions (and optional targets) to plot-ready arrays.

    - If input is list[Tensor|ndarray] → returns stacked or list arrays (auto fallback)
    - If input is list[dict] → returns dict of stacked/list arrays by key
    - If input is a dict of lists → stacks along batch where possible
    - Targets, when matching keys, are added under "targets" alongside predictions
    """
    # Common case: list of batches
    if isinstance(preds, list):
        try:
            return stack_batches(preds, mode="stack")
        except NotStackableError:
            return stack_batches(preds, mode="list")

    # Dict of lists or arrays
    if isinstance(preds, Mapping):
        result: dict[str, Any] = {}
        for k, v in preds.items():
            key = str(k)
            if isinstance(v, list):
                try:
                    result[key] = stack_batches(v, mode="stack")
                except NotStackableError:
                    result[key] = stack_batches(v, mode="list")
            else:
                result[key] = to_numpy(v)
        # Add targets if provided
        if targets is not None:
            if isinstance(targets, Mapping):
                result["targets"] = {k: to_numpy(v) for k, v in targets.items()}
            else:
                # Assume targets is an array-like object
                result["tgt/y"] = to_numpy(targets)
        return result

    # Single object → numpy
    return to_numpy(preds)
