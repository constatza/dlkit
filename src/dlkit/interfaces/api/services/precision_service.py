"""Precision service.

Re-exports from tools.config.precision for backward compatibility.
All new code should import directly from dlkit.tools.config.precision.
"""

from dlkit.tools.config.precision.service import PrecisionService, get_precision_service

__all__ = ["PrecisionService", "get_precision_service"]
