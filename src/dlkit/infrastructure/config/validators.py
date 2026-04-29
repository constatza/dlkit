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

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .workflow_configs import (
        InferenceWorkflowConfig,
        OptimizationWorkflowConfig,
        TrainingWorkflowConfig,
    )

from .data_entries import PathBasedEntry, PathFeature, PathTarget, ValueBasedEntry
from .enums import DatasetFamily

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
    """Validate a single PathFeature/PathTarget entry has a valid existing path."""
    if isinstance(entry, PathFeature) and entry.path is not None:
        path = _coerce_path(entry.path)
        if path is not None:
            _assert_path_exists(path, f"{role} #{index + 1}")
    elif isinstance(entry, PathTarget) and entry.path is not None:
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
    if dataset.family in (DatasetFamily.GRAPH, DatasetFamily.TIMESERIES):
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


def validate_training_config_complete(config: TrainingWorkflowConfig) -> None:
    """Validate that training config has all required sections for component building.

    This validator ensures the config is ready for BuildFactory.build_components().
    It checks:
    1. Required sections are present (DATAMODULE, DATASET, MODEL)
    2. DATASET has at least one feature or target (new config format only)
    3. All feature/target paths exist (if provided)

    Args:
        config: Training workflow configuration

    Raises:
        ConfigValidationError: If config is incomplete or invalid

    Example:
        ```python
        config = TrainingWorkflowConfig.model_validate(toml_dict)
        config = config.patch({"DATASET": dataset_settings})

        # Validate before building
        validate_training_config_complete(config)  # Raises if incomplete

        # Safe to build components
        components = BuildFactory().build_components(config)
        ```
    """
    _check_required_sections(config, ["DATAMODULE", "DATASET", "MODEL"], "training")

    # Only FlexibleDataset configs require features/targets; graph/timeseries use own schema.
    if config.DATASET is not None:
        _validate_flexible_dataset_entries(config.DATASET)

        # Validate feature paths exist and placeholders are resolved
        for i, feature in enumerate(config.DATASET.features):
            _validate_entry_has_data(feature, i, "Feature")

        # Validate target paths exist and placeholders are resolved
        for i, target in enumerate(config.DATASET.targets):
            _validate_entry_has_data(target, i, "Target")

    # Validate MODEL checkpoint path if provided
    if config.MODEL is not None and config.MODEL.checkpoint is not None:
        checkpoint_path = _coerce_path(config.MODEL.checkpoint)
        if checkpoint_path is not None:
            _assert_path_exists(checkpoint_path, "Model checkpoint")


def validate_inference_config_complete(config: InferenceWorkflowConfig) -> None:
    """Validate that inference config has all required sections.

    This validator ensures the config is ready for inference execution. It checks:
    1. SESSION.workflow == "inference"
    2. MODEL.checkpoint is provided and exists
    3. For batch inference: DATAMODULE and DATASET present with valid paths

    Args:
        config: Inference workflow configuration

    Raises:
        ConfigValidationError: If config is incomplete or invalid

    Example:
        ```python
        config = InferenceWorkflowConfig.model_validate(toml_dict)

        # Validate before inference
        validate_inference_config_complete(config)

        # Safe to run inference
        predictor = load_model(config.MODEL.checkpoint)
        ```
    """
    # Validate inference mode enabled
    if config.SESSION.workflow != "inference":
        raise ConfigValidationError(
            "SESSION.workflow must be 'inference' for inference workflows. "
            "Add 'workflow = \"inference\"' under [SESSION] section."
        )

    # Validate checkpoint provided and exists
    if config.MODEL.checkpoint is None:
        raise ConfigValidationError(
            "MODEL.checkpoint is required for inference. "
            "Add 'checkpoint = \"/path/to/model.ckpt\"' under [MODEL] section."
        )

    checkpoint_path = _coerce_path(config.MODEL.checkpoint)
    if checkpoint_path is not None:
        _assert_path_exists(checkpoint_path, "Model checkpoint")

    # Validate batch inference config if provided
    if config.DATAMODULE is not None or config.DATASET is not None:
        # Both must be present for batch inference
        if config.DATAMODULE is None:
            raise ConfigValidationError(
                "DATAMODULE is required when DATASET is provided for batch inference. "
                "Add [DATAMODULE] section or remove [DATASET] section."
            )
        if config.DATASET is None:
            raise ConfigValidationError(
                "DATASET is required when DATAMODULE is provided for batch inference. "
                "Add [DATASET] section or remove [DATAMODULE] section."
            )

        # Validate dataset has features
        if not config.DATASET.features:
            raise ConfigValidationError(
                "DATASET must have at least one feature for batch inference. "
                "Add [[DATASET.features]] sections to your config."
            )

        # Validate feature paths
        for i, feature in enumerate(config.DATASET.features):
            _validate_path_entry(feature, i, "Feature")


def validate_optimization_config_complete(config: OptimizationWorkflowConfig) -> None:
    """Validate that optimization config has all required sections.

    This validator ensures the config is ready for hyperparameter optimization. It checks:
    1. Required sections present (DATAMODULE, DATASET, MODEL)
    2. OPTUNA.enabled is True
    3. OPTUNA.model dict has parameter ranges
    4. DATASET has valid data
    5. All paths exist

    Args:
        config: Optimization workflow configuration

    Raises:
        ConfigValidationError: If config is incomplete or invalid

    Example:
        ```python
        config = OptimizationWorkflowConfig.model_validate(toml_dict)

        # Validate before optimization
        validate_optimization_config_complete(config)

        # Safe to run optimization
        study = optimization_service.execute_optimization(config)
        ```
    """
    # Validate Optuna enabled
    if not config.OPTUNA.enabled:
        raise ConfigValidationError(
            "OPTUNA.enabled must be true for optimization workflows. "
            "Add 'enabled = true' under [OPTUNA] section."
        )

    # Validate Optuna has parameter ranges
    if not config.OPTUNA.model:
        raise ConfigValidationError(
            "OPTUNA.model must define hyperparameter search spaces. "
            "Add [OPTUNA.model] section with parameter ranges. "
            "Example: [OPTUNA.model]\nlr = [0.0001, 0.01]\nbatch_size = [16, 32, 64]"
        )

    _check_required_sections(config, ["DATAMODULE", "DATASET", "MODEL"], "optimization")

    # DATASET must have at least one feature or target
    if config.DATASET is not None:
        if not config.DATASET.features and not config.DATASET.targets:
            raise ConfigValidationError(
                "DATASET must have at least one feature or target for optimization. "
                "Add [[DATASET.features]] or [[DATASET.targets]] sections."
            )

        # Validate paths
        for i, feature in enumerate(config.DATASET.features):
            _validate_path_entry(feature, i, "Feature")

        for i, target in enumerate(config.DATASET.targets):
            _validate_path_entry(target, i, "Target")


# Convenience function for workflow auto-detection
def validate_config_complete(
    config: TrainingWorkflowConfig | InferenceWorkflowConfig | OptimizationWorkflowConfig,
) -> None:
    """Validate config completeness based on workflow type.

    Auto-detects workflow type and calls appropriate validator.

    Args:
        config: Workflow configuration (any type)

    Raises:
        ConfigValidationError: If config is incomplete or invalid
        TypeError: If config type is not recognized
    """
    from .workflow_configs import InferenceWorkflowConfig, OptimizationWorkflowConfig

    if isinstance(config, OptimizationWorkflowConfig):
        validate_optimization_config_complete(config)
    elif isinstance(config, InferenceWorkflowConfig):
        validate_inference_config_complete(config)
    else:
        validate_training_config_complete(config)


def validate_runtime_preflight(
    config: TrainingWorkflowConfig | InferenceWorkflowConfig | OptimizationWorkflowConfig,
) -> list[str]:
    """Return a list of preflight error messages (empty list means OK).

    Checks file existence, checkpoint paths, and other runtime-only conditions
    that cannot be validated at parse time.

    Args:
        config: Workflow configuration to check

    Returns:
        List of error message strings; empty if all checks pass
    """
    from .workflow_configs import (
        InferenceWorkflowConfig,
        OptimizationWorkflowConfig,
        TrainingWorkflowConfig,
    )

    errors: list[str] = []

    try:
        match config:
            case InferenceWorkflowConfig():
                validate_inference_config_complete(config)
            case OptimizationWorkflowConfig():
                validate_optimization_config_complete(config)
            case TrainingWorkflowConfig():
                validate_training_config_complete(config)
    except ConfigValidationError as e:
        errors.append(str(e))

    return errors
