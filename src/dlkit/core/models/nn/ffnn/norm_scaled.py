from __future__ import annotations

from typing import Literal
from collections.abc import Callable

import torch
from torch import Tensor, nn

from dlkit.core.models.nn.base import ShapeAwareModel
from dlkit.core.models.nn.ffnn.simple import ConstantWidthFFNN
from dlkit.core.shape_specs import IShapeSpec, NullShapeSpec


class NormScaledFFNN(ShapeAwareModel):
    """Wrap a base FFNN with input/output norm scaling.

    This module enforces homogeneous scaling consistency for Ax = b by:
      1) Normalizing b: b_scaled = b / ||b||_p
      2) Predicting x_scaled = base_model(b_scaled)
      3) Rescaling x: x = x_scaled * ||b||_p

    Precision is managed by PyTorch Lightning's precision plugins via the Trainer.

    Args:
        base_model: Underlying module that operates on normalized inputs.
        unified_shape: Shape specification describing (b, x) dimensions.
        norm: Which vector norm to use for scaling; one of {"l2", "l1", "linf"}.
        eps_gain: Multiplier for machine epsilon used to avoid division by zero.
        keep_stats: When True, :meth:`forward` returns ``(x, {"norm": norms})``.
    """

    SUPPORTED_NORMS = {"l2", "l1", "linf"}
    DEFAULT_NORM = "l2"
    DEFAULT_EPS_GAIN = 10.0

    def __init__(
        self,
        *,
        base_model: nn.Module | None = None,
        unified_shape: IShapeSpec,
        norm: str = DEFAULT_NORM,
        eps_gain: float = DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
    ) -> None:
        if base_model is None:
            msg = (
                "NormScaledFFNN cannot be used directly. "
                "Choose a concrete wrapper such as NormScaledLinearFFNN or NormScaledConstantWidthFFNN."
            )
            raise ValueError(msg)
        if not isinstance(base_model, nn.Module):
            msg = "base_model must be an instance of torch.nn.Module"
            raise TypeError(msg)
        if norm not in self.SUPPORTED_NORMS:
            raise ValueError(f"norm must be one of {self.SUPPORTED_NORMS}, got {norm!r}.")
        if eps_gain <= 0:
            raise ValueError("eps_gain must be > 0.")

        self.norm = norm
        self.eps_gain = float(eps_gain)
        self.keep_stats = keep_stats

        pending_base_model = base_model

        super().__init__(unified_shape=unified_shape)

        # Register the base model once the nn.Module initialization completed
        self.base_model = pending_base_model

    def accepts_shape(self, shape_spec: IShapeSpec) -> bool:
        if isinstance(shape_spec, NullShapeSpec):
            return False
        input_shape = shape_spec.get_input_shape()
        output_shape = shape_spec.get_output_shape()

        if input_shape is None or output_shape is None:
            return False

        if len(input_shape) != 1 or len(output_shape) != 1:
            return False

        if input_shape[0] <= 0 or output_shape[0] <= 0:
            return False

        return True

    @staticmethod
    def _compute_eps(x: Tensor, gain: float) -> float:
        finfo = torch.finfo(x.dtype)
        return float(gain * finfo.eps)

    def _vector_norm(self, x: Tensor) -> Tensor:
        if self.norm == "l2":
            return torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True)
        if self.norm == "l1":
            return torch.linalg.vector_norm(x, ord=1, dim=-1, keepdim=True)
        return torch.linalg.vector_norm(x, ord=float("inf"), dim=-1, keepdim=True)

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        # Input casting is now handled by Lightning's precision plugin
        b = x
        if not torch.is_floating_point(b):
            raise TypeError(f"Expected floating point tensor, received dtype={b.dtype}.")
        if b.ndim < 1:
            raise ValueError(
                f"Expected b to have at least 1 dimension, got shape {tuple(b.shape)}."
            )

        norms = self._vector_norm(b)
        eps = self._compute_eps(b, self.eps_gain)
        safe_div = norms.clamp_min(eps)

        b_scaled = b / safe_div
        x_scaled = self.base_model(b_scaled)
        x = x_scaled * norms

        if self.keep_stats:
            return x, {"norm": norms}
        return x


class NormScaledLinearFFNN(NormScaledFFNN):
    """NormScaledFFNN with a single Linear layer as the base network.

    Precision is managed by PyTorch Lightning's precision plugins via the Trainer.
    """

    def __init__(
        self,
        *,
        unified_shape: IShapeSpec,
        bias: bool = True,
        norm: str = NormScaledFFNN.DEFAULT_NORM,
        eps_gain: float = NormScaledFFNN.DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
    ) -> None:
        input_shape = unified_shape.get_input_shape()
        output_shape = unified_shape.get_output_shape()
        if input_shape is None or output_shape is None:
            raise ValueError("NormScaledLinearFFNN requires both input and output shapes.")
        if len(input_shape) != 1 or len(output_shape) != 1:
            raise ValueError("NormScaledLinearFFNN only supports 1D feature shapes.")

        base_model = nn.Linear(input_shape[0], output_shape[0], bias=bias)
        super().__init__(
            base_model=base_model,
            unified_shape=unified_shape,
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )


class NormScaledConstantWidthFFNN(NormScaledFFNN):
    """NormScaledFFNN backed by ConstantWidthFFNN.

    Precision is managed by PyTorch Lightning's precision plugins via the Trainer.
    """

    def __init__(
        self,
        *,
        unified_shape: IShapeSpec,
        hidden_size: int,
        num_layers: int,
        norm: str = NormScaledFFNN.DEFAULT_NORM,
        eps_gain: float = NormScaledFFNN.DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        activation_fn = activation if activation is not None else nn.functional.gelu
        base_model = ConstantWidthFFNN(
            unified_shape=unified_shape,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation_fn,
            normalize=normalize,  # type: ignore[arg-type]
            dropout=dropout,
        )
        super().__init__(
            base_model=base_model,
            unified_shape=unified_shape,
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )
