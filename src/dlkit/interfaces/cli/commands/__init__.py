"""CLI command modules for DLKit."""

# Import command modules to make them available
from . import config, predict, optimize, server, train

__all__ = [
    "train",
    "predict",
    "optimize",
    "config",
    "server",
]
