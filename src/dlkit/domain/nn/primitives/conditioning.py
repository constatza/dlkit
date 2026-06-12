from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch.nn as nn
from torch import Tensor


@runtime_checkable
class IConditionedModule(Protocol):
    """Protocol for modules that accept a conditioning tensor alongside the primary input.

    Any module implementing ``forward(x, condition)`` satisfies this protocol.
    """

    def forward(self, x: Tensor, condition: Tensor) -> Tensor: ...


class AsConditioned(nn.Module):
    """Adapt an unconditional ``nn.Module`` to the ``IConditionedModule`` interface.

    ``forward(x, condition)`` calls ``self.module(x)``, silently discarding the
    conditioning tensor. Useful for mixing conditioned and unconditional blocks
    inside a ``ConditionedSequential``.

    Args:
        module (nn.Module): The unconditional module to wrap.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        """Forward pass ignoring ``condition``.

        Args:
            x (Tensor): Primary input tensor.
            condition (Tensor): Conditioning tensor (ignored).

        Returns:
            Tensor: Output of the wrapped module applied to ``x``.
        """
        return self.module(x)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation (FiLM) conditioning layer.

    Applies the affine transformation ``(1 + γ(c)) * x + β(c)`` where
    ``γ`` and ``β`` are learned linear projections from the condition vector.

    Both projections are **zero-initialised** (weights and biases), so at
    initialisation ``γ(c) = 0`` and ``β(c) = 0``, giving
    ``forward(x, c) == x`` for any input. This identity initialisation
    prevents the conditioning signal from disrupting a pre-trained or
    randomly-initialised backbone at the start of training.

    Op chain:
        1. ``gamma = to_gamma(condition)``  — shape ``(..., feature_dim)``
        2. ``beta  = to_beta(condition)``   — shape ``(..., feature_dim)``
        3. return ``(1 + gamma) * x + beta``

    Args:
        condition_dim (int): Dimensionality of the conditioning vector.
        feature_dim (int): Dimensionality of the features to modulate.
    """

    def __init__(self, condition_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.to_gamma = nn.Linear(condition_dim, feature_dim)
        self.to_beta = nn.Linear(condition_dim, feature_dim)
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        """Apply FiLM modulation.

        Args:
            x (Tensor): Features to modulate, shape ``(..., feature_dim)``.
            condition (Tensor): Conditioning vector, shape ``(..., condition_dim)``.

        Returns:
            Tensor: Modulated features, shape ``(..., feature_dim)``.
        """
        gamma = self.to_gamma(condition)
        beta = self.to_beta(condition)
        return (1.0 + gamma) * x + beta


class ConditionedSequential(nn.Module):
    """Sequential chain of conditioned blocks, each receiving the same condition.

    Each block must satisfy ``IConditionedModule`` (i.e. expose
    ``forward(x, condition) -> Tensor``). The condition tensor is passed
    unchanged to every block.

    Op chain:
        ``x = block_0(x, condition); x = block_1(x, condition); ...``

    Args:
        *blocks (IConditionedModule): Ordered conditioned blocks to apply.
    """

    def __init__(self, *blocks: IConditionedModule) -> None:
        super().__init__()
        self.blocks: nn.ModuleList = nn.ModuleList(list(blocks))  # ty: ignore[invalid-argument-type]

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        """Pass ``x`` through every block, injecting ``condition`` at each step.

        Args:
            x (Tensor): Primary input tensor.
            condition (Tensor): Conditioning tensor forwarded to every block.

        Returns:
            Tensor: Output after the full conditioned chain.
        """
        for block in self.blocks:
            x = block(x, condition)  # type: ignore[arg-type]
        return x


class ConditionedResidualSequential(ConditionedSequential):
    """Conditioned sequential with an end-to-end skip connection.

    Computes ``output = body(x, condition) + shortcut(x)`` where ``body``
    is the inherited ``ConditionedSequential`` chain.

    When ``shortcut=None``, an identity skip is used; this requires the
    input and output dimensions of the full chain to match.

    Args:
        *blocks (IConditionedModule): Ordered conditioned blocks forming the body.
        shortcut (nn.Module | None): Optional skip-path projection. ``None`` for identity.
    """

    def __init__(self, *blocks: IConditionedModule, shortcut: nn.Module | None = None) -> None:
        super().__init__(*blocks)
        self.shortcut = shortcut

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        """Apply conditioned body and add the skip connection.

        Args:
            x (Tensor): Primary input tensor.
            condition (Tensor): Conditioning tensor forwarded to every block.

        Returns:
            Tensor: ``body(x, condition) + shortcut(x)`` (or ``+ x`` if no shortcut).
        """
        out = super().forward(x, condition)
        skip = self.shortcut(x) if self.shortcut is not None else x
        return out + skip


__all__ = [
    "AsConditioned",
    "ConditionedResidualSequential",
    "ConditionedSequential",
    "FiLMLayer",
    "IConditionedModule",
]
