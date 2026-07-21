"""Global RNG seeding for DLKit.

This module provides the single, centralized entry point for seeding
Python/NumPy/PyTorch global RNG state.
"""

from .service import apply_global_seed

__all__ = [
    "apply_global_seed",
]
