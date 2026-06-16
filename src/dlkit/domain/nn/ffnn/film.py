from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Self

import torch.nn as nn
from torch import Tensor

from dlkit.domain.nn.contracts import TabulaRSpec
from dlkit.domain.nn.primitives.conditioning import (
    ConditionedResidualSequential,
    FiLMLayer,
    IConditionedModule,
)
from dlkit.domain.nn.primitives.dense import DenseBlock
from dlkit.domain.nn.types import ActivationName, NormalizerName
from dlkit.domain.nn.utils import resolve_activation


class FiLMBlock(IConditionedModule):
    """Dense block followed by FiLM modulation.

    Op chain: ``Norm → Act → Lin(in→out) → Drop → FiLM(γ·h + β)``

    Args:
        in_features (int): Input feature dimension.
        out_features (int): Output feature dimension.
        condition_dim (int): Condition vector dimension.
        activation (ActivationName | Callable | None): Activation name or callable (default: relu).
        normalize (NormalizerName | None): Norm layer name or None.
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        condition_dim: int,
        *,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: NormalizerName | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        activation = resolve_activation(activation)
        self.dense = DenseBlock(
            in_features, out_features, activation=activation, normalize=normalize, dropout=dropout
        )
        self.film = FiLMLayer(condition_dim, out_features)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        """Apply dense block then FiLM modulation.

        Args:
            x (Tensor): Input tensor, shape ``(..., in_features)``.
            condition (Tensor): Conditioning vector, shape ``(..., condition_dim)``.

        Returns:
            Tensor: Modulated output, shape ``(..., out_features)``.
        """
        return self.film(self.dense(x), condition)


class FiLMResidualBlock(IConditionedModule):
    """Two dense blocks + FiLM modulation + residual skip.

    Op chain: ``[Norm→Act→Lin→Drop] × 2 → FiLM((1+γ)·h+β) → h + x``

    Square-only: ``in_features == out_features``. Skip is identity.

    Args:
        feature_dim (int): Feature dimension (in == out).
        condition_dim (int): Condition vector dimension.
        activation (ActivationName | Callable | None): Activation name or callable.
        normalize (NormalizerName | None): Norm layer name or None.
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        feature_dim: int,
        condition_dim: int,
        *,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: NormalizerName | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        activation = resolve_activation(activation)
        self.block1 = DenseBlock(
            feature_dim, feature_dim, activation=activation, normalize=normalize, dropout=dropout
        )
        self.block2 = DenseBlock(
            feature_dim, feature_dim, activation=activation, normalize=normalize, dropout=dropout
        )
        self.film = FiLMLayer(condition_dim, feature_dim)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        """Apply two dense blocks, FiLM modulation, and residual addition.

        Args:
            x (Tensor): Input tensor, shape ``(..., feature_dim)``.
            condition (Tensor): Conditioning vector, shape ``(..., condition_dim)``.

        Returns:
            Tensor: Output with residual skip, shape ``(..., feature_dim)``.
        """
        h = self.block2(self.block1(x))
        return x + self.film(h, condition)


class VarWidthFiLMFFNN(nn.Module):
    """Variable-width FiLM-conditioned feedforward network.

    Op chain:
        ``Lin(in→l₀) → [FiLMBlock(lᵢ→lᵢ₊₁)] × N → Lin(lₙ→out)``

    Pre-activation style: the embedding Linear has no activation;
    the first FiLMBlock's DenseBlock supplies the first Norm→Act.

    Note:
        At least two elements in ``layers`` are needed for any FiLM conditioning
        to occur; a single-element list produces ``Lin → Lin`` with no conditioned
        blocks.

    Args:
        in_features (int): Input dimension.
        out_features (int): Output dimension.
        condition_dim (int): Condition vector dimension.
        layers (Sequence[int]): Hidden-state widths, e.g. ``[128, 64, 64]``.
        activation (ActivationName | Callable | None): Activation for DenseBlocks.
        normalize (NormalizerName | None): Norm layer name or None.
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        condition_dim: int,
        layers: Sequence[int],
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: NormalizerName | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(layers) < 2:
            raise ValueError(
                "VarWidthFiLMFFNN requires at least two elements in `layers` to produce "
                f"FiLM-conditioned hidden blocks; got {len(layers)}."
            )
        widths = list(layers)
        self.num_layers = len(widths) - 1
        self.embed = nn.Linear(in_features, widths[0])
        self.hidden: nn.ModuleList = nn.ModuleList(
            [
                FiLMBlock(
                    widths[i],
                    widths[i + 1],
                    condition_dim,
                    activation=activation,
                    normalize=normalize,
                    dropout=dropout,
                )
                for i in range(len(widths) - 1)
            ]
        )
        self.head = nn.Linear(widths[-1], out_features)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        """Embed, apply FiLM blocks, then project to output.

        Args:
            x (Tensor): Input tensor, shape ``(..., in_features)``.
            condition (Tensor): Conditioning vector, shape ``(..., condition_dim)``.

        Returns:
            Tensor: Output tensor, shape ``(..., out_features)``.
        """
        x = self.embed(x)
        for block in self.hidden:
            x = block(x, condition)
        return self.head(x)

    @classmethod
    def from_contract(
        cls,
        contract: TabulaRSpec,
        condition_dim: int,
        **kwargs: Any,
    ) -> Self:
        """Construct from a TabulaRSpec contract.

        Args:
            contract (TabulaRSpec): Shape contract with in_shape and out_shape.
            condition_dim (int): Condition vector dimension.
            **kwargs: Passed to __init__ (layers, activation, etc.)

        Returns:
            Self: Constructed instance.
        """
        return cls(
            in_features=contract.in_shape[0],
            out_features=contract.out_shape[0],
            condition_dim=condition_dim,
            **kwargs,
        )


class FiLMEmbeddedFFNN(nn.Module):
    """Embedded FiLM-conditioned network with constant-width residual body.

    Op chain:
        ``Lin(in→H) → ConditionedResidualSequential(FiLMResidualBlock×N) + E2E skip → Lin(H→out)``

    Nested residuals: per-block skip inside each FiLMResidualBlock
    plus an end-to-end skip across the whole body.

    Args:
        in_features (int): Input dimension.
        out_features (int): Output dimension.
        condition_dim (int): Condition vector dimension.
        hidden_size (int): Constant hidden width H.
        num_layers (int): Number of FiLMResidualBlocks in the body.
        activation (ActivationName | Callable | None): Activation for DenseBlocks.
        normalize (NormalizerName | None): Norm layer or None.
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        condition_dim: int,
        hidden_size: int,
        num_layers: int,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: NormalizerName | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1.")
        self.embed = nn.Linear(in_features, hidden_size)
        self.body: ConditionedResidualSequential = ConditionedResidualSequential(
            *[
                FiLMResidualBlock(
                    hidden_size,
                    condition_dim,
                    activation=activation,
                    normalize=normalize,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Linear(hidden_size, out_features)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        """Embed, pass through residual body, then project to output.

        Args:
            x (Tensor): Input tensor, shape ``(..., in_features)``.
            condition (Tensor): Conditioning vector, shape ``(..., condition_dim)``.

        Returns:
            Tensor: Output tensor, shape ``(..., out_features)``.
        """
        x = self.embed(x)
        x = self.body(x, condition)
        return self.head(x)

    @classmethod
    def from_contract(
        cls,
        contract: TabulaRSpec,
        condition_dim: int,
        **kwargs: Any,
    ) -> Self:
        """Construct from a TabulaRSpec contract.

        Args:
            contract (TabulaRSpec): Shape contract with in_shape and out_shape.
            condition_dim (int): Condition vector dimension.
            **kwargs: Passed to __init__ (hidden_size, num_layers, etc.)

        Returns:
            Self: Constructed instance.
        """
        return cls(
            in_features=contract.in_shape[0],
            out_features=contract.out_shape[0],
            condition_dim=condition_dim,
            **kwargs,
        )


class FiLMFFNN(VarWidthFiLMFFNN):
    """Constant-width FiLM-conditioned feedforward network.

    Op chain:
        ``Lin(in→H) → [FiLMBlock(H→H)] × N → Lin(H→out)``

    This mirrors :class:`dlkit.domain.nn.ffnn.residual.FFNN` but keeps the FiLM
    conditioned hidden transitions non-residual.

    Args:
        in_features (int): Input dimension.
        out_features (int): Output dimension.
        condition_dim (int): Condition vector dimension.
        hidden_size (int): Width of all hidden states.
        num_layers (int): Number of hidden ``FiLMBlock`` transitions.
        activation (ActivationName | Callable | None): Activation for DenseBlocks.
        normalize (NormalizerName | None): Norm layer name or None.
        dropout (float): Dropout rate.

    Note:
        ``num_layers`` must be at least 1 so at least one FiLM-conditioned
        hidden transition exists.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        condition_dim: int,
        hidden_size: int,
        num_layers: int,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: NormalizerName | None = None,
        dropout: float = 0.0,
    ) -> None:
        if num_layers < 1:
            raise ValueError("FiLMFFNN requires num_layers >= 1.")
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            condition_dim=condition_dim,
            layers=[hidden_size] * (num_layers + 1),
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )

    @classmethod
    def from_contract(
        cls,
        contract: TabulaRSpec,
        condition_dim: int,
        **kwargs: Any,
    ) -> Self:
        """Construct from a TabulaRSpec contract."""
        return cls(
            in_features=contract.in_shape[0],
            out_features=contract.out_shape[0],
            condition_dim=condition_dim,
            **kwargs,
        )


__all__ = [
    "FiLMBlock",
    "FiLMEmbeddedFFNN",
    "FiLMFFNN",
    "FiLMResidualBlock",
    "VarWidthFiLMFFNN",
]
