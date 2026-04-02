"""Precision strategy configuration for DLKit settings.

This module provides precision strategy enumeration, context management,
and service coordination for configuration purposes.
"""

from .context import (
    PrecisionContext,
    PrecisionProvider,
    current_precision_override,
    get_global_precision_context,
    get_precision_context,
    precision_override,
)
from .service import PrecisionService, get_precision_service
from .strategy import PrecisionStrategy

__all__ = [
    "PrecisionContext",
    "PrecisionProvider",
    "PrecisionService",
    "PrecisionStrategy",
    "current_precision_override",
    "get_global_precision_context",
    "get_precision_context",
    "get_precision_service",
    "precision_override",
]
