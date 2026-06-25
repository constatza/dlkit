"""Completeness validators for workflow configurations.

This module provides pre-build validation functions that check whether all required
configuration sections are present before component construction. These validators
separate "config syntax is valid" (Pydantic) from "config is ready to use" (completeness).

Design Pattern: Specification Pattern + Fail-Fast Validation
- Each workflow has a dedicated completeness validator
- Validators check required sections are present (not None)
- Validators perform cross-section validation (e.g., path existence, field dependencies)
- Clear error messages guide users to fix configuration issues

Architecture Principles:
- SOLID compliant: Single Responsibility (validation), Open/Closed (extensible)
- Fail-fast: Raise before expensive component construction
- Type-safe: Type annotations make requirements explicit
- Composable: Validators can be chained or combined
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from .data_entries import PathBasedEntry, ValueBasedEntry
from .entry_base import DataEntry
from .enums import DatasetFamily

if TYPE_CHECKING:
    from .job_config import InferenceJobConfig, JobConfig, SearchJobConfig, TrainingJobConfig


def _validate_entry_paths(entries: Iterable[DataEntry], role_label: str) -> None:
    """Raise ValueError if any path-based entry has a path that does not exist.

    Args:
        entries: Entries to validate.
        role_label: Human-readable label for error messages (e.g. "Feature", "Target").

    Raises:
        ValueError: If a path-based entry's path does not exist on disk.
    """
    for entry in entries:
        if isinstance(entry, PathBasedEntry) and entry.path is not None and not entry.path.exists():
            raise ValueError(f"{role_label} path does not exist: {entry.path}")


_NON_FLEXIBLE_DATASET_NAMES: frozenset[str] = frozenset({"GraphDataset", "TimeSeriesDataset"})


class ConfigValidationError(ValueError):
    """Raised when configuration is incomplete or invalid for workflow execution."""

    def __init__(
        self,
        message: str,
        model_class: str = "",
        section_data: dict[str, str] | None = None,
    ):
        super().__init__(message)
        self.model_class = model_class
        self.section_data = section_data or {}


# ============================================================================
# Private Helper Functions (DRY rule extraction)
# ============================================================================


def _coerce_path(value: str | Path | None) -> Path | None:
    """Coerce a string or Path to Path, returning None if input is None."""
    if value is None:
        return None
    return Path(value) if isinstance(value, str) else value


def _assert_path_exists(path: Path, label: str) -> None:
    """Raise ConfigValidationError if path does not exist."""
    if not path.exists():
        raise ConfigValidationError(f"{label} path does not exist: {path}.")


def _validate_path_entry(entry: object, index: int, role: str) -> None:
    """Validate a single PathBasedEntry has a valid existing path."""
    if isinstance(entry, PathBasedEntry) and entry.path is not None:
        path = _coerce_path(entry.path)
        if path is not None:
            _assert_path_exists(path, f"{role} #{index + 1}")


def _validate_entry_has_data(entry: object, index: int, role: str) -> None:
    """Validate an entry has either a valid path or an in-memory value."""
    if isinstance(entry, PathBasedEntry) and not entry.has_path():
        raise ConfigValidationError(
            f"{role} #{index + 1} is a placeholder without path/value: "
            f"{getattr(entry, 'name', 'unknown')}"
        )
    _validate_path_entry(entry, index, role)
    if isinstance(entry, ValueBasedEntry) and not entry.has_value():
        raise ConfigValidationError(
            f"{role} #{index + 1} is missing in-memory data: {getattr(entry, 'name', 'unknown')}"
        )


def _validate_flexible_dataset_entries(dataset: object) -> None:
    """Raise ConfigValidationError if a FlexibleDataset has no features or targets."""
    from .dataset_settings import DatasetSettings

    if not isinstance(dataset, DatasetSettings):
        return
    if dataset.family in (DatasetFamily.GRAPH,):
        return
    if getattr(dataset, "name", None) in _NON_FLEXIBLE_DATASET_NAMES:
        return
    if not (dataset.features or dataset.targets):
        raise ConfigValidationError(
            "DATASET must have at least one feature or target. "
            "Add [[DATASET.features]] or [[DATASET.targets]] sections to your config."
        )


def _check_required_sections(config: object, sections: list[str], workflow: str) -> None:
    """Raise ConfigValidationError listing any missing sections."""
    missing = [s for s in sections if getattr(config, s, None) is None]
    if missing:
        raise ConfigValidationError(
            f"Missing required sections for {workflow}: {', '.join(missing)}. "
            "These sections must be provided in TOML config or injected programmatically "
            "before calling build_components()."
        )


def validate_training_config_complete(config: TrainingJobConfig) -> None:
    """Validate that training config has all required sections for component building.

    This validator ensures the config is ready for BuildFactory.build_components().
    It checks:
    1. data section is present with at least one feature or target
    2. All feature/target paths exist (if provided)
    3. model.checkpoint path exists if provided

    Args:
        config: Training job configuration

    Raises:
        ConfigValidationError: If config is incomplete or invalid

    Example:
        ```python
        config = TrainingJobConfig.model_validate(toml_dict)

        # Validate before building
        validate_training_config_complete(config)  # Raises if incomplete

        # Safe to build components
        components = BuildFactory().build_components(config)
        ```
    """
    _validate_job_config_data(config)


def validate_inference_config_complete(config: InferenceJobConfig) -> None:
    """Validate that inference config has all required sections.

    This validator ensures the config is ready for inference execution. It checks:
    1. model.checkpoint is provided and exists
    2. For batch inference: data present with valid paths

    Args:
        config: Inference job configuration

    Raises:
        ConfigValidationError: If config is incomplete or invalid

    Example:
        ```python
        config = InferenceJobConfig.model_validate(toml_dict)

        # Validate before inference
        validate_inference_config_complete(config)

        # Safe to run inference
        predictor = load_model(config.model.checkpoint)
        ```
    """
    _validate_inference_job_config(config)


def validate_optimization_config_complete(config: SearchJobConfig) -> None:
    """Validate that optimization config has all required sections.

    This validator ensures the config is ready for hyperparameter optimization. It checks:
    1. data section is present with valid entries
    2. All paths exist
    3. SearchJobConfig.space is non-empty (validated by model validator)

    Args:
        config: Search job configuration

    Raises:
        ConfigValidationError: If config is incomplete or invalid

    Example:
        ```python
        config = SearchJobConfig.model_validate(toml_dict)

        # Validate before optimization
        validate_optimization_config_complete(config)

        # Safe to run optimization
        study = optimization_service.execute_optimization(config)
        ```
    """
    _validate_job_config_data(config)


# Convenience function for workflow auto-detection
def validate_config_complete(
    config: TrainingJobConfig | SearchJobConfig | InferenceJobConfig | JobConfig,
) -> None:
    """Validate config completeness based on workflow type.

    Auto-detects workflow type and calls the appropriate JobConfig validator.

    Args:
        config: Job configuration

    Raises:
        ConfigValidationError: If config is incomplete or invalid
        TypeError: If config type is not recognized
    """
    from .job_config import InferenceJobConfig, JobConfig, SearchJobConfig, TrainingJobConfig

    if isinstance(config, SearchJobConfig):
        _validate_search_job_config(config)
        return
    if isinstance(config, InferenceJobConfig):
        _validate_inference_job_config(config)
        return
    if isinstance(config, (TrainingJobConfig, JobConfig)):
        _validate_job_config_data(config)
        return

    raise TypeError(f"Unsupported config type: {type(config).__name__}")


def _validate_search_job_config(config: SearchJobConfig) -> None:
    """Validate a new-style SearchJobConfig."""
    _validate_job_config_data(config)


def _validate_job_config_data(config: JobConfig) -> None:
    """Validate data entries in a new-style JobConfig."""
    if config.data is None:
        return
    ds = config.data
    for i, feature in enumerate(ds.features):
        _validate_entry_has_data(feature, i, "Feature")
    for i, target in enumerate(ds.targets):
        _validate_entry_has_data(target, i, "Target")
    if config.model is not None and config.model.checkpoint is not None:
        checkpoint_path = _coerce_path(config.model.checkpoint)
        if checkpoint_path is not None:
            _assert_path_exists(checkpoint_path, "Model checkpoint")


def _validate_inference_job_config(config: InferenceJobConfig) -> None:
    """Validate a new-style InferenceJobConfig."""
    if config.model.checkpoint is None:
        raise ConfigValidationError(
            "model.checkpoint is required for inference. "
            "Add 'checkpoint = \"/path/to/model.ckpt\"' under [model] section."
        )
    checkpoint_path = _coerce_path(config.model.checkpoint)
    if checkpoint_path is not None:
        _assert_path_exists(checkpoint_path, "Model checkpoint")


def validate_runtime_preflight(
    config: TrainingJobConfig | SearchJobConfig | InferenceJobConfig | JobConfig,
) -> list[str]:
    """Return a list of preflight error messages (empty list means OK).

    Checks file existence, checkpoint paths, and other runtime-only conditions
    that cannot be validated at parse time.

    Args:
        config: Job configuration to check

    Returns:
        List of error message strings; empty if all checks pass
    """
    from .job_config import InferenceJobConfig, SearchJobConfig, TrainingJobConfig

    errors: list[str] = []

    try:
        match config:
            case SearchJobConfig():
                _validate_search_job_config(config)
            case InferenceJobConfig():
                _validate_inference_job_config(config)
            case TrainingJobConfig():
                _validate_job_config_data(config)
            case _:
                _validate_job_config_data(config)
    except ConfigValidationError as e:
        errors.append(str(e))

    return errors
