"""Thin structural protocols for settings contracts.

Defines the minimal interface surface needed by consumers outside the infrastructure
layer. Infrastructure types satisfy these protocols via structural subtyping — no
explicit inheritance required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class SessionSettingsProtocol(Protocol):
    """Minimal session contract — exposes the fields accessed through the abstract type."""

    @property
    def name(self) -> str: ...

    @property
    def workflow(self) -> str: ...


@runtime_checkable
class OptunaSettingsProtocol(Protocol):
    """Minimal Optuna contract — exposes the fields accessed through the abstract type."""

    @property
    def enabled(self) -> bool: ...


@runtime_checkable
class ModelSettingsProtocol(Protocol):
    """Minimal model contract — exposes the fields accessed through the abstract type."""

    @property
    def checkpoint(self) -> Path | str | None: ...


@runtime_checkable
class BaseSettingsProtocol(Protocol):
    """Minimal contract shared by all DLKit workflow configs.

    Return types use concrete sub-protocols where callers access sub-fields through
    this type; otherwise ``object`` is used since consumers only perform None checks.
    """

    @property
    def SESSION(self) -> SessionSettingsProtocol | None: ...

    @property
    def MODEL(self) -> ModelSettingsProtocol | None: ...

    @property
    def DATAMODULE(self) -> object: ...

    @property
    def DATASET(self) -> object: ...

    @property
    def PATHS(self) -> object: ...

    @property
    def EXTRAS(self) -> object: ...

    @property
    def is_training(self) -> bool: ...

    @property
    def is_inference(self) -> bool: ...

    @property
    def has_data_config(self) -> bool: ...

    def get_datamodule_config(self) -> object: ...

    def get_dataset_config(self) -> object: ...


@runtime_checkable
class TrainingSettingsProtocol(BaseSettingsProtocol, Protocol):
    """Extends BaseSettingsProtocol with training-workflow attributes."""

    @property
    def TRAINING(self) -> object: ...

    @property
    def MLFLOW(self) -> object: ...

    @property
    def OPTUNA(self) -> OptunaSettingsProtocol | None: ...

    @property
    def mlflow_enabled(self) -> bool: ...

    @property
    def optuna_enabled(self) -> bool: ...

    @property
    def has_training_config(self) -> bool: ...

    def get_training_config(self) -> object: ...
