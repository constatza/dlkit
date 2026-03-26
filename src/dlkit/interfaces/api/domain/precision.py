"""Precision domain objects and protocols.

Re-exports from tools.config.precision for backward compatibility.
All new code should import directly from dlkit.tools.config.precision.
"""

from dlkit.tools.config.precision.context import (
    PrecisionContext,
    PrecisionProvider,
    current_precision_override,
    get_global_precision_context,
    get_precision_context,
    precision_override,
)
from dlkit.tools.config.precision.strategy import PrecisionStrategy

__all__ = [
    "PrecisionContext",
    "PrecisionProvider",
    "PrecisionStrategy",
    "current_precision_override",
    "get_global_precision_context",
    "get_precision_context",
    "precision_override",
]
