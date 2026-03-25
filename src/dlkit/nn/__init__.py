"""Compatibility package alias for ``dlkit.core.models.nn``."""

from __future__ import annotations

from dlkit.core.models import nn as _nn

__path__ = _nn.__path__
__all__ = getattr(_nn, "__all__", [])
