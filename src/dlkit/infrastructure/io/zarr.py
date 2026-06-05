"""Backward-compat shim — canonical location is ``dlkit.infrastructure.zarr``."""

from dlkit.infrastructure.zarr import ILazyReader, ZarrLazyReader

__all__ = ["ILazyReader", "ZarrLazyReader"]
