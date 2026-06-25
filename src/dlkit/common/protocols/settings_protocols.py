"""Thin structural protocols for settings contracts.

Defines the minimal interface surface needed by consumers outside the infrastructure
layer. Infrastructure types satisfy these protocols via structural subtyping — no
explicit inheritance required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class ModelSettingsProtocol(Protocol):
    """Minimal model contract — exposes the fields accessed through the abstract type."""

    @property
    def checkpoint(self) -> Path | str | None: ...


@runtime_checkable
class BaseSettingsProtocol(Protocol):
    """Minimal contract shared by all DLKit workflow configs.

    Satisfied by BaseWorkflowSettings (UPPERCASE fields). JobConfig satisfies a
    different structural shape — callers that need JobConfig should type-check
    against JobConfig directly.
    """

    @property
    def MODEL(self) -> ModelSettingsProtocol | None: ...

    @property
    def is_training(self) -> bool: ...

    @property
    def is_inference(self) -> bool: ...

    @property
    def has_data_config(self) -> bool: ...


@runtime_checkable
class TrainingSettingsProtocol(BaseSettingsProtocol, Protocol):
    """Extends BaseSettingsProtocol with training-workflow attributes."""

    @property
    def TRAINING(self) -> object: ...

    @property
    def mlflow_enabled(self) -> bool: ...

    @property
    def has_training_config(self) -> bool: ...

    def get_training_config(self) -> object: ...
