"""Runtime data packages and data-facing services."""

from .geometry import infer_geometry
from .splits import SplitDataset

__all__ = ["SplitDataset", "infer_geometry"]
