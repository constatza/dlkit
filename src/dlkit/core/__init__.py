"""Core modeling, data, and training primitives for DLKit.

This package exposes the primary sub-packages that contain the
restructured implementation after the domain split:

- :mod:`dlkit.core.datamodules`
- :mod:`dlkit.core.datasets`
- :mod:`dlkit.core.datatypes`
- :mod:`dlkit.core.models`
- :mod:`dlkit.core.training`

Historically we eagerly imported these modules to help certain frozen or
zip-import scenarios. That eager import, however, introduced circular
dependencies during application bootstrap when configuration utilities need
lightweight data typing support. We now rely on Python's normal package
loading and advertise the submodules via ``__all__`` instead of importing
them eagerly.
"""

from collections.abc import Iterable
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

__all__ = ["datamodules", "datasets", "datatypes", "models", "training"]

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from . import datamodules as datamodules
    from . import datasets as datasets
    from . import datatypes as datatypes
    from . import models as models
    from . import training as training


def __getattr__(name: str) -> ModuleType:
    """Dynamically import core subpackages on first access.

    This preserves the lightweight import behaviour that the new architecture
    expects while keeping the public API intact for callers that relied on the
    previous eager imports.
    """
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> Iterable[str]:
    """Ensure dir() lists lazily imported attributes."""
    return sorted(set(__all__) | set(globals().keys()))
