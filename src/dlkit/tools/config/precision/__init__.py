"""Precision strategy configuration for DLKit settings.

This module provides only the core precision strategy enumeration for
configuration purposes. Business logic has been moved to domain services.
"""

from .strategy import PrecisionStrategy

__all__ = ["PrecisionStrategy"]
