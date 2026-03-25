"""Shared utilities for building neural network components.

These are pure functions with no side effects, intended to be imported by
primitives, encoders, and higher-level model modules.
"""

from __future__ import annotations

import torch
from torch import nn

from dlkit.core.datatypes.networks import NormalizerName


def make_norm_layer(
    normalize: NormalizerName | None,
    features: int,
    timesteps: int | None = None,
) -> nn.Module:
    """Instantiate a normalization layer from a string identifier.

    Uses match/case dispatch — adding a new normalizer only requires a new case here.

    Args:
        normalize: Normalizer name or None / "none" for no normalization.
        features: Channel / feature count (used by all normalizer types).
        timesteps: Sequence length; required only when ``normalize == "layer"``
            and the input is 3-D (channels × timesteps).

    Returns:
        An ``nn.Module`` ready to be stored in the parent module.

    Raises:
        ValueError: If *normalize* is not a recognised identifier.
    """
    match normalize:
        case None | "none":
            return nn.Identity()
        case "layer":
            shape: int | list[int] = [features, timesteps] if timesteps is not None else features
            return nn.LayerNorm(shape)
        case "batch":
            return nn.BatchNorm1d(features)
        case "instance":
            return nn.InstanceNorm1d(features)
        case _:
            raise ValueError(f"Unsupported normalizer: {normalize!r}")


def build_channel_schedule(start: int, end: int, steps: int) -> list[int]:
    """Return a linearly spaced integer list of length *steps* from *start* to *end*.

    Replaces the repeated ``torch.linspace(a, b, n).int().tolist()`` pattern
    used when constructing progressive channel/timestep schedules.

    Args:
        start: First value (inclusive).
        end: Last value (inclusive).
        steps: Total number of values (including start and end).

    Returns:
        A list of *steps* integers linearly spaced between *start* and *end*.
    """
    return torch.linspace(start, end, steps).int().tolist()
