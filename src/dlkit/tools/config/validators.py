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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .workflow_configs import (
        TrainingWorkflowConfig,
        InferenceWorkflowConfig,
        OptimizationWorkflowConfig,
    )

from .data_entries import PathFeature, PathTarget, PathBasedEntry, ValueBasedEntry


class ConfigValidationError(ValueError):
    """Raised when configuration is incomplete or invalid for workflow execution."""

    pass


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
        config = config.model_copy(update={"DATASET": dataset_settings})

        # Validate before building
        validate_training_config_complete(config)  # Raises if incomplete

        # Safe to build components
        components = BuildFactory().build_components(config)
        ```
    """
    from . import GeneralSettings

    missing_sections = []

    # Check required sections present
    if config.DATAMODULE is None:
        missing_sections.append("DATAMODULE")
    if config.DATASET is None:
        missing_sections.append("DATASET")
    if config.MODEL is None:
        missing_sections.append("MODEL")

    if missing_sections:
        raise ConfigValidationError(
            f"Missing required sections for training: {', '.join(missing_sections)}. "
            "These sections must be provided in TOML config or injected programmatically "
            "before calling build_components()."
        )

    # DATASET must have at least one feature or target (new format only)
    # Legacy GeneralSettings may use x/y or other dataset construction patterns
    if config.DATASET is not None and not isinstance(config, GeneralSettings):
        if not config.DATASET.features and not config.DATASET.targets:
            raise ConfigValidationError(
                "DATASET must have at least one feature or target. "
                "Add [[DATASET.features]] or [[DATASET.targets]] sections to your config."
            )

        # Validate feature paths exist and placeholders are resolved
        from pathlib import Path
        for i, feature in enumerate(config.DATASET.features):
            if isinstance(feature, PathBasedEntry) and not feature.has_path():
                raise ConfigValidationError(
                    f"Feature #{i+1} is a placeholder without path/value: {feature.name or 'unknown'}"
                )
            if isinstance(feature, PathFeature) and feature.path is not None:
                # Handle both str and Path types (model_construct may bypass coercion)
                path = Path(feature.path) if isinstance(feature.path, str) else feature.path
                if not path.exists():
                    raise ConfigValidationError(
                        f"Feature #{i+1} path does not exist: {path}. "
                        f"Ensure the path is correct or the file has been created."
                    )
            if isinstance(feature, ValueBasedEntry) and not feature.has_value():
                raise ConfigValidationError(
                    f"Feature #{i+1} is missing in-memory data: {feature.name or 'unknown'}"
                )

        # Validate target paths exist and placeholders are resolved
        for i, target in enumerate(config.DATASET.targets):
            if isinstance(target, PathBasedEntry) and not target.has_path():
                raise ConfigValidationError(
                    f"Target #{i+1} is a placeholder without path/value: {target.name or 'unknown'}"
                )
            if isinstance(target, PathTarget) and target.path is not None:
                # Handle both str and Path types (model_construct may bypass coercion)
                path = Path(target.path) if isinstance(target.path, str) else target.path
                if not path.exists():
                    raise ConfigValidationError(
                        f"Target #{i+1} path does not exist: {path}. "
                        f"Ensure the path is correct or the file has been created."
                    )
            if isinstance(target, ValueBasedEntry) and not target.has_value():
                raise ConfigValidationError(
                    f"Target #{i+1} is missing in-memory data: {target.name or 'unknown'}"
                )

    # Validate MODEL checkpoint path if provided
    if config.MODEL is not None and config.MODEL.checkpoint is not None:
        # Handle both str and Path types
        from pathlib import Path
        checkpoint_path = Path(config.MODEL.checkpoint) if isinstance(config.MODEL.checkpoint, str) else config.MODEL.checkpoint
        if not checkpoint_path.exists():
            raise ConfigValidationError(
                f"Model checkpoint does not exist: {checkpoint_path}. "
                "Remove the checkpoint field to train from scratch, or provide a valid path."
            )


def validate_inference_config_complete(config: InferenceWorkflowConfig) -> None:
    """Validate that inference config has all required sections.

    This validator ensures the config is ready for inference execution. It checks:
    1. SESSION.inference is True
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
        predictor = load_predictor(config.MODEL.checkpoint)
        ```
    """
    # Validate inference mode enabled
    if not config.SESSION.inference:
        raise ConfigValidationError(
            "SESSION.inference must be true for inference workflows. "
            "Add 'inference = true' under [SESSION] section."
        )

    # Validate checkpoint provided and exists
    if config.MODEL.checkpoint is None:
        raise ConfigValidationError(
            "MODEL.checkpoint is required for inference. "
            "Add 'checkpoint = \"/path/to/model.ckpt\"' under [MODEL] section."
        )

    # Handle both str and Path types
    from pathlib import Path
    checkpoint_path = Path(config.MODEL.checkpoint) if isinstance(config.MODEL.checkpoint, str) else config.MODEL.checkpoint
    if not checkpoint_path.exists():
        raise ConfigValidationError(
            f"Model checkpoint does not exist: {checkpoint_path}. "
            "Ensure the checkpoint path is correct and the file exists."
        )

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
        from pathlib import Path
        for i, feature in enumerate(config.DATASET.features):
            if isinstance(feature, PathFeature) and feature.path is not None:
                # Handle both str and Path types (model_construct may bypass coercion)
                path = Path(feature.path) if isinstance(feature.path, str) else feature.path
                if not path.exists():
                    raise ConfigValidationError(
                        f"Feature #{i+1} path does not exist: {path}. "
                        "Ensure the path is correct or the file has been created."
                    )


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

    # Check required sections present (same as training)
    missing_sections = []
    if config.DATAMODULE is None:
        missing_sections.append("DATAMODULE")
    if config.DATASET is None:
        missing_sections.append("DATASET")
    if config.MODEL is None:
        missing_sections.append("MODEL")

    if missing_sections:
        raise ConfigValidationError(
            f"Missing required sections for optimization: {', '.join(missing_sections)}. "
            "Hyperparameter optimization requires complete training configuration."
        )

    # DATASET must have at least one feature or target
    if config.DATASET is not None:
        if not config.DATASET.features and not config.DATASET.targets:
            raise ConfigValidationError(
                "DATASET must have at least one feature or target for optimization. "
                "Add [[DATASET.features]] or [[DATASET.targets]] sections."
            )

        # Validate paths
        from pathlib import Path
        for i, feature in enumerate(config.DATASET.features):
            if isinstance(feature, PathFeature) and feature.path is not None:
                # Handle both str and Path types (model_construct may bypass coercion)
                path = Path(feature.path) if isinstance(feature.path, str) else feature.path
                if not path.exists():
                    raise ConfigValidationError(
                        f"Feature #{i+1} path does not exist: {path}."
                    )

        for i, target in enumerate(config.DATASET.targets):
            if isinstance(target, PathTarget) and target.path is not None:
                # Handle both str and Path types (model_construct may bypass coercion)
                path = Path(target.path) if isinstance(target.path, str) else target.path
                if not path.exists():
                    raise ConfigValidationError(
                        f"Target #{i+1} path does not exist: {path}."
                    )


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
    from .workflow_configs import (
        TrainingWorkflowConfig,
        InferenceWorkflowConfig,
        OptimizationWorkflowConfig,
    )
    from .workflow_settings import (
        TrainingWorkflowSettings,
        InferenceWorkflowSettings,
    )
    from . import GeneralSettings

    if isinstance(config, OptimizationWorkflowConfig):
        validate_optimization_config_complete(config)
    elif isinstance(config, InferenceWorkflowConfig):
        validate_inference_config_complete(config)
    elif isinstance(config, TrainingWorkflowConfig):
        validate_training_config_complete(config)
    elif isinstance(config, InferenceWorkflowSettings):
        validate_inference_config_complete(config)
    elif isinstance(config, TrainingWorkflowSettings):
        validate_training_config_complete(config)
    elif isinstance(config, GeneralSettings):
        # Legacy GeneralSettings: detect workflow type and validate
        # For backwards compatibility, treat as training workflow
        # (most common case and safest assumption)
        validate_training_config_complete(config)
    else:
        raise TypeError(
            f"Unknown config type: {type(config).__name__}. "
            "Expected TrainingWorkflowConfig, InferenceWorkflowConfig, OptimizationWorkflowConfig, or GeneralSettings."
        )
