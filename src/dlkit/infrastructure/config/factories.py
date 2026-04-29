"""Settings factory for efficient partial loading following SOLID principles."""

from __future__ import annotations

import tomllib
from pathlib import Path

from dlkit.infrastructure.config.validators import ConfigValidationError

from .workflow_configs import (
    InferenceWorkflowConfig,
    OptimizationWorkflowConfig,
    TrainingWorkflowConfig,
)
from .workflow_types import WorkflowConfig


class WorkflowSettingsLoader:
    """Factory class for config loading.

    Dispatches on SESSION.workflow to return the appropriate workflow config type.
    """

    def load_settings(self, config_path: Path | str) -> WorkflowConfig:
        """Load workflow config from TOML, dispatching on SESSION.workflow.

        Args:
            config_path: Path to TOML configuration file.

        Returns:
            WorkflowConfig: The appropriate workflow config type based on SESSION.workflow.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValueError: If SESSION.workflow is unknown.
            pydantic.ValidationError: If validation fails.
        """
        from dlkit.infrastructure.config.core.patching import patch_model
        from dlkit.infrastructure.config.core.sources import DLKitTomlSource, _read_env_patches

        path = Path(config_path)

        # Minimal read to determine workflow mode without full validation.
        # Parse failures here must surface clearly instead of silently falling
        # back to the training discriminator.
        try:
            raw = tomllib.loads(path.read_text())
            mode = raw.get("SESSION", {}).get("workflow", "train")
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise ConfigValidationError(
                f"Failed to read workflow discriminator from {path}: {exc}",
                model_class="WorkflowSettingsLoader",
                section_data={"config_path": str(path)},
            ) from exc

        # Load full TOML as a dict
        source = DLKitTomlSource(path)
        toml_data = source()

        match mode:
            case "train":
                config = TrainingWorkflowConfig.model_validate(toml_data)
            case "optimize":
                config = OptimizationWorkflowConfig.model_validate(toml_data)
            case "inference":
                config = InferenceWorkflowConfig.model_validate(toml_data)
            case _:
                raise ValueError(
                    f"Unknown SESSION.workflow value: {mode!r}. "
                    "Expected 'train', 'optimize', or 'inference'."
                )

        # Apply environment variable patches if present
        if env := _read_env_patches("DLKIT_", "__"):
            config = patch_model(config, env)

        # Propagate SESSION.root_dir to the global environment if set
        from dlkit.infrastructure.config.environment import sync_session_root_to_environment

        sync_session_root_to_environment(config)

        return config


# Default factory instance for convenience
default_settings_loader = WorkflowSettingsLoader()


# Convenience function that delegates to the default loader


def load_settings(config_path: Path | str) -> WorkflowConfig:
    """Load workflow configuration from TOML file with automatic dispatching.

    This is the primary API for loading DLKit configurations. It inspects
    SESSION.workflow to determine the appropriate workflow config type and
    returns the correctly-typed config.

    **Dispatching Logic:**
    - If SESSION.workflow = "train": Returns TrainingWorkflowConfig
    - If SESSION.workflow = "optimize": Returns OptimizationWorkflowConfig
    - If SESSION.workflow = "inference": Returns InferenceWorkflowConfig
    - If unspecified: Defaults to "train"

    **Loaded sections (varies by workflow type):**

    Training:
        - Required: SESSION, TRAINING
        - Optional: DATAMODULE, DATASET, MODEL, MLFLOW, OPTUNA, PATHS, EXTRAS

    Optimization:
        - Required: SESSION, TRAINING, OPTUNA
        - Optional: DATAMODULE, DATASET, MODEL, MLFLOW, PATHS, EXTRAS

    Inference:
        - Required: SESSION, MODEL
        - Optional: DATAMODULE, DATASET, PATHS, EXTRAS

    Args:
        config_path: Path to TOML configuration file.

    Returns:
        WorkflowConfig: One of TrainingWorkflowConfig, OptimizationWorkflowConfig,
                       or InferenceWorkflowConfig based on SESSION.workflow.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If SESSION.workflow is unknown or validation fails.

    Examples:
        >>> from dlkit.infrastructure.config import load_settings
        >>> config = load_settings("train_config.toml")
        >>> if isinstance(config, TrainingWorkflowConfig):
        ...     print("Training mode")

    See Also:
        - `TrainingWorkflowConfig`: For training-specific configuration
        - `OptimizationWorkflowConfig`: For optimization-specific configuration
        - `InferenceWorkflowConfig`: For inference-specific configuration
    """
    return default_settings_loader.load_settings(config_path)


def load_sections(
    config_path: Path | str, sections: list[str], *, strict: bool = False
) -> WorkflowConfig:
    """Load specific configuration sections for custom workflows.

    For most workflows, use `load_settings()` instead. This function is for
    advanced use cases requiring specific section combinations. It still dispatches
    on SESSION.workflow to return the appropriate workflow config type.

    Args:
        config_path: Path to TOML configuration file.
        sections: List of section names to load (e.g., ["MODEL", "DATASET"]).
        strict: If True, all specified sections must exist; if False (default),
               missing sections are ignored.

    Returns:
        WorkflowConfig: Appropriate workflow config type based on SESSION.workflow.

    Raises:
        ValueError: If no valid sections specified or unknown sections requested.
        FileNotFoundError: If config file doesn't exist.

    Examples:
        >>> settings = load_sections("config.toml", ["MODEL", "DATASET"])
        >>>
        >>> # Strict loading ensures all sections exist
        >>> settings = load_sections("config.toml", ["MODEL", "DATASET"], strict=True)

    See Also:
        - `load_settings()`: Load full workflow configuration (recommended).
    """
    if not sections:
        raise ValueError("At least one section must be specified")

    path = Path(config_path)

    if strict:
        import tomllib

        with open(path, "rb") as handle:
            data = tomllib.load(handle)
        available = [key.upper() for key in data if isinstance(key, str)]
        missing = [s for s in sections if s not in available]
        if missing:
            raise ValueError(f"Strict mode: Required sections missing from config file: {missing}")

    # Still dispatch based on SESSION.workflow
    return default_settings_loader.load_settings(path)
