from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from dlkit.common.shapes import InputShapes, OutputShapes

DEFAULT_SCALE_EQUIVARIANT_NORM = "l2"
DEFAULT_SCALE_EQUIVARIANT_EPS_GAIN = 10.0


class _NormScalingBase(nn.Module):
    """Base class owning norm validation, eps computation, and vector-norm logic.

    Subclasses implement ``forward`` with their own signature.

    Args:
        base_model (nn.Module): The wrapped module to apply after normalisation.
        norm (str): Norm type; one of ``SUPPORTED_NORMS``.
        eps_gain (float): Multiplier applied to ``finfo.eps`` to form the safe-division floor.
        keep_stats (bool): When ``True``, ``forward`` returns ``(output, {"norm": norms})``.
    """

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
        """Return a dtype-appropriate epsilon floor.

        Args:
            x (Tensor): Tensor whose dtype determines machine epsilon.
            gain (float): Multiplier applied to ``torch.finfo(x.dtype).eps``.

        Returns:
            float: Safe-division floor value.
        """
        finfo = torch.finfo(x.dtype)
        return float(gain * finfo.eps)

    def _vector_norm(self, x: Tensor) -> Tensor:
        """Compute the per-sample vector norm along the last dimension.

        Args:
            x (Tensor): Input tensor of any shape with at least one dimension.

        Returns:
            Tensor: Norm values with the last dimension kept (``keepdim=True``).
        """
        match self.norm:
            case "l2":
                return torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True)
            case "l1":
                return torch.linalg.vector_norm(x, ord=1, dim=-1, keepdim=True)
            case _:
                return torch.linalg.vector_norm(x, ord=float("inf"), dim=-1, keepdim=True)

    def _validated_norms(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Validate x and return (norms, safe_div).

        Args:
            x: Input tensor.

        Returns:
            Tuple of (norms, safe_div) where safe_div = norms.clamp_min(eps).

        Raises:
            TypeError: If x is not floating point.
            ValueError: If x has fewer than 1 dimension.
        """
        if not torch.is_floating_point(x):
            raise TypeError(f"Expected floating point tensor, received dtype={x.dtype}.")
        if x.ndim < 1:
            raise ValueError(
                f"Expected x to have at least 1 dimension, got shape {tuple(x.shape)}."
            )
        norms = self._vector_norm(x)
        return norms, norms.clamp_min(self._compute_eps(x, self.eps_gain))


class ScaleEquivariantWrapper(_NormScalingBase):
    """Wrap a base module with norm-based input/output scaling.

    Applies the transformation ``base_model(x / ||x||) * ||x||`` so that the
    module is scale-equivariant: ``f(α x) = α f(x)`` for any scalar ``α > 0``.

    Args:
        base_model (nn.Module): Module to wrap.
        norm (str): Norm type; one of ``{"l2", "l1", "linf"}``.
        eps_gain (float): Safe-division floor multiplier (relative to ``finfo.eps``).
        keep_stats (bool): If ``True``, return ``(output, {"norm": norms})``.
    """

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Normalise ``x``, pass through ``base_model``, then rescale.

        Args:
            x (Tensor): Floating-point input with at least one dimension.

        Returns:
            Tensor | tuple[Tensor, dict[str, Tensor]]: Scaled output, or
            ``(output, {"norm": norms})`` when ``keep_stats=True``.

        Raises:
            TypeError: If ``x`` is not a floating-point tensor.
            ValueError: If ``x`` has no dimensions.
        """
        norms, safe_div = self._validated_norms(x)
        x_scaled = self.base_model(x / safe_div) * norms

        if self.keep_stats:
            return x_scaled, {"norm": norms}
        return x_scaled


class ConditionedScaleEquivariantWrapper(_NormScalingBase):
    """Scale-equivariant wrapper for conditioned modules accepting ``(x, condition)``.

    Applies ``base_model(x / ||x||, condition) * ||x||`` so that the wrapped
    module is scale-equivariant in ``x`` while still receiving an external
    conditioning signal.

    Args:
        base_model (nn.Module): Conditioned module with signature
            ``forward(x, condition)``.
        norm (str): Norm type; one of ``{"l2", "l1", "linf"}``.
        eps_gain (float): Safe-division floor multiplier (relative to ``finfo.eps``).
        keep_stats (bool): If ``True``, return ``(output, {"norm": norms})``.
    """

    def forward(self, x: Tensor, condition: Tensor) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Normalise ``x``, pass ``(x_norm, condition)`` through ``base_model``, rescale.

        Args:
            x (Tensor): Floating-point input with at least one dimension.
            condition (Tensor): Conditioning tensor forwarded as-is to ``base_model``.

        Returns:
            Tensor | tuple[Tensor, dict[str, Tensor]]: Scaled output, or
            ``(output, {"norm": norms})`` when ``keep_stats=True``.

        Raises:
            TypeError: If ``x`` is not a floating-point tensor.
            ValueError: If ``x`` has no dimensions.
        """
        norms, safe_div = self._validated_norms(x)
        out = self.base_model(x / safe_div, condition) * norms
        return (out, {"norm": norms}) if self.keep_stats else out


def shape_aware_kwargs(
    input_shapes: InputShapes,
    output_shapes: OutputShapes,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Merge explicit kwargs with shape-derived in/out feature sizes.

    Args:
        input_shapes: Mapping from feature entry name to its shape.
        output_shapes: Mapping from target entry name to its shape.
        kwargs: Additional keyword arguments; in_features/out_features are stripped.

    Returns:
        A dict with in_features and out_features from the shapes, plus remaining kwargs.
    """
    filtered = {k: v for k, v in kwargs.items() if k not in ("in_features", "out_features")}
    return {
        "in_features": next(iter(input_shapes.values()))[0],
        "out_features": next(iter(output_shapes.values()))[0],
        **filtered,
    }


__all__ = [
    "DEFAULT_SCALE_EQUIVARIANT_EPS_GAIN",
    "DEFAULT_SCALE_EQUIVARIANT_NORM",
    "ConditionedScaleEquivariantWrapper",
    "ScaleEquivariantWrapper",
    "shape_aware_kwargs",
]
