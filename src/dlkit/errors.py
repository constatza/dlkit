"""User-facing error namespace.

Thin re-exports from ``dlkit.common.errors`` so users can write::

    from dlkit.errors import DLKitError, WorkflowError

instead of the internal path::

    from dlkit.common.errors import DLKitError, WorkflowError
"""

from dlkit.common.errors import (
    ConfigurationError,
    DLKitError,
    ModelLoadingError,
    ModelStateError,
    PluginError,
    StrategyError,
    WorkflowError,
)

__all__ = [
    "DLKitError",
    "ConfigurationError",
    "WorkflowError",
    "StrategyError",
    "ModelStateError",
    "ModelLoadingError",
    "PluginError",
]
