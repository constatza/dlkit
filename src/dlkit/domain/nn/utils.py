"""Shared utilities for building neural network components.

These are pure functions with no side effects, intended to be imported by
primitives, encoders, and higher-level model modules.
"""

from __future__ import annotations

from collections.abc import Callable
from math import exp, log
from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch import nn

from dlkit.domain.nn.types import ActivationName, NormalizerName

type WidthScheduleMode = Literal["linear", "geometric"]


def _identity(x: torch.Tensor) -> torch.Tensor:
    """Identity activation function that returns input unchanged."""
    return x


def resolve_activation(
    name: ActivationName | str | Callable | None,
    default: ActivationName = "relu",
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Resolve an activation function from a name, callable, or None.

    Args:
        name: Activation name, callable, or None.
        default: Activation name used when ``name`` is None.

    Returns:
        The resolved activation callable.
    """
    if callable(name):
        return cast(Callable[[torch.Tensor], torch.Tensor], name)
    if name is None:
        name = default
    match name:
        case "none" | "identity":
            return _identity
        case "relu":
            return F.relu
        case "gelu":
            return F.gelu
        case "silu":
            return F.silu
        case "tanh":
            return torch.tanh
        case "sigmoid":
            return torch.sigmoid
        case "leaky_relu":
            return F.leaky_relu
        case _:
            raise ValueError(f"Unsupported activation: {name!r}")


def resolved_activation_name(
    name: ActivationName | str | Callable | None,
    default: ActivationName = "relu",
) -> str:
    """Return the human-readable name resolve_activation() resolved to.

    Args:
        name: Activation name, callable, or None (mirrors resolve_activation's input).
        default: Activation name used when ``name`` is None.

    Returns:
        The resolved activation name, for logging/introspection purposes.
    """
    if callable(name):
        return getattr(name, "__name__", repr(name))
    return name if name is not None else default


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


def build_width_schedule(
    start: int,
    end: int,
    steps: int,
    *,
    mode: WidthScheduleMode = "geometric",
    round_to: int | None = None,
) -> list[int]:
    """Return an integer feature-width schedule between exact endpoint widths.

    ``"geometric"`` spaces widths multiplicatively, mirroring the encoder/decoder
    convention of changing feature capacity by a near-constant ratio per stage.
    ``"linear"`` spaces widths additively.

    Args:
        start: First width (inclusive).
        end: Last width (inclusive).
        steps: Total number of widths, including endpoints.
        mode: Spacing rule for intermediate widths.
        round_to: Optional positive multiple for intermediate widths.

    Returns:
        A list of *steps* positive integer widths with exact first/last values.

    Raises:
        ValueError: If widths, steps, mode, or ``round_to`` are invalid.
    """
    if start <= 0 or end <= 0:
        raise ValueError("start and end widths must be positive")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if round_to is not None and round_to <= 0:
        raise ValueError("round_to must be positive when provided")
    if steps == 1:
        return [start]

    match mode:
        case "linear":
            raw = [start + (end - start) * i / (steps - 1) for i in range(steps)]
        case "geometric":
            start_log = log(start)
            end_log = log(end)
            raw = [exp(start_log + (end_log - start_log) * i / (steps - 1)) for i in range(steps)]
        case _:
            raise ValueError(f"Unsupported width schedule mode: {mode!r}")

    widths = [_round_width(value, round_to=round_to) for value in raw]
    widths[0] = start
    widths[-1] = end
    return widths


def _round_width(value: float, *, round_to: int | None) -> int:
    width = max(1, round(value))
    if round_to is None:
        return width
    return max(round_to, round(width / round_to) * round_to)
