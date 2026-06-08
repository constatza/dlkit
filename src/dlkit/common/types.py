"""Common type aliases used across layers."""

from typing import Literal

type NormalizerName = Literal["batch", "layer", "instance", "none"]
type ActivationName = Literal[
    "relu", "gelu", "silu", "tanh", "sigmoid", "leaky_relu", "none", "identity"
]

__all__ = ["NormalizerName", "ActivationName"]
