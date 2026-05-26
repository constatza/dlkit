from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from dlkit.domain.nn.contracts import TabulaRSpec

DEFAULT_SCALE_EQUIVARIANT_NORM = "l2"
DEFAULT_SCALE_EQUIVARIANT_EPS_GAIN = 10.0


class ScaleEquivariantWrapper(nn.Module):
    """Wrap a base module with norm-based input/output scaling."""

    SUPPORTED_NORMS = {"l2", "l1", "linf"}

    def __init__(
        self,
        *,
        base_model: nn.Module,
        norm: str = DEFAULT_SCALE_EQUIVARIANT_NORM,
        eps_gain: float = DEFAULT_SCALE_EQUIVARIANT_EPS_GAIN,
        keep_stats: bool = False,
    ) -> None:
        if not isinstance(base_model, nn.Module):
            raise TypeError("base_model must be an instance of torch.nn.Module")
        if norm not in self.SUPPORTED_NORMS:
            raise ValueError(f"norm must be one of {self.SUPPORTED_NORMS}, got {norm!r}.")
        if eps_gain <= 0:
            raise ValueError("eps_gain must be > 0.")

        super().__init__()
        self.base_model = base_model
        self.norm = norm
        self.eps_gain = float(eps_gain)
        self.keep_stats = keep_stats

    @staticmethod
    def _compute_eps(x: Tensor, gain: float) -> float:
        finfo = torch.finfo(x.dtype)
        return float(gain * finfo.eps)

    def _vector_norm(self, x: Tensor) -> Tensor:
        match self.norm:
            case "l2":
                return torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True)
            case "l1":
                return torch.linalg.vector_norm(x, ord=1, dim=-1, keepdim=True)
            case _:
                return torch.linalg.vector_norm(x, ord=float("inf"), dim=-1, keepdim=True)

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if not torch.is_floating_point(x):
            raise TypeError(f"Expected floating point tensor, received dtype={x.dtype}.")
        if x.ndim < 1:
            raise ValueError(
                f"Expected x to have at least 1 dimension, got shape {tuple(x.shape)}."
            )

        norms = self._vector_norm(x)
        eps = self._compute_eps(x, self.eps_gain)
        safe_div = norms.clamp_min(eps)
        x_scaled = self.base_model(x / safe_div) * norms

        if self.keep_stats:
            return x_scaled, {"norm": norms}
        return x_scaled


def contract_aware_kwargs(contract: TabulaRSpec, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Merge explicit kwargs with contract-derived in/out feature sizes.

    Args:
        contract: A TabulaRSpec providing in_shape and out_shape.
        kwargs: Additional keyword arguments; in_features/out_features are stripped.

    Returns:
        A dict with in_features and out_features from the contract, plus remaining kwargs.
    """
    filtered = {k: v for k, v in kwargs.items() if k not in ("in_features", "out_features")}
    return {
        "in_features": contract.in_shape[0],
        "out_features": contract.out_shape[0],
        **filtered,
    }


__all__ = [
    "DEFAULT_SCALE_EQUIVARIANT_EPS_GAIN",
    "DEFAULT_SCALE_EQUIVARIANT_NORM",
    "ScaleEquivariantWrapper",
    "contract_aware_kwargs",
]
