"""User-facing registry namespace.

Curated re-exports for registration decorators and registry introspection.
"""

from dlkit.infrastructure.registry import (
    RegistryEntry,
    describe_model,
    list_registered_datasets,
    list_registered_models,
    register_datamodule,
    register_dataset,
    register_loss,
    register_metric,
    register_model,
)

__all__ = [
    "RegistryEntry",
    "describe_model",
    "list_registered_datasets",
    "list_registered_models",
    "register_datamodule",
    "register_dataset",
    "register_loss",
    "register_metric",
    "register_model",
]
