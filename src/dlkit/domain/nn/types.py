"""Low-level neural-network type aliases."""

from typing import Literal

type NormalizerName = Literal["batch", "layer", "instance", "none"]

__all__ = ["NormalizerName"]
