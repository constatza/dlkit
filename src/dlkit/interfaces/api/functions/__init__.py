"""API functions module."""

from .core import train, optimize
from .config import validate_config, generate_template, validate_template
from .execution import execute

__all__ = [
    # Core workflow functions
    "train",
    "optimize",
    # Configuration functions
    "validate_config",
    "generate_template",
    "validate_template",
    # Unified execution function
    "execute",
]
