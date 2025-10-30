"""Settings factory for efficient partial loading following SOLID principles."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar, cast, Any, Type

from .workflow_settings import (
    BaseWorkflowSettings,
    TrainingWorkflowSettings,
    # BREAKING CHANGE: InferenceWorkflowSettings removed
)
from .session_settings import SessionSettings
from .datamodule_settings import DataModuleSettings
from .dataset_settings import DatasetSettings
from .training_settings import TrainingSettings as TrainingConfig
from .components.model_components import ModelComponentSettings
from .mlflow_settings import MLflowSettings
from .optuna_settings import OptunaSettings
from .paths_settings import PathsSettings
from .extras_settings import ExtrasSettings

T = TypeVar("T", bound=BaseWorkflowSettings)


class PartialSettingsLoader:
    """Factory class for efficient partial config loading.

    Implements SettingsLoaderProtocol and follows the Factory Method pattern
    to create workflow-specific settings with minimal parsing overhead.

    This class follows:
    - Single Responsibility: Only handles settings creation
    - Open/Closed: Extensible for new workflow types
    - Dependency Inversion: Depends on abstractions (protocols)
    - Interface Segregation: Provides minimal, focused methods
    """

    def __init__(self) -> None:
        """Initialize the partial settings loader."""
        # Section mapping for workflow types
        self._base_required_sections = {
            "SESSION": SessionSettings,
            "DATAMODULE": DataModuleSettings,
            "DATASET": DatasetSettings,
        }

        self._base_optional_sections = {
            "MODEL": ModelComponentSettings,
            "PATHS": PathsSettings,
            "EXTRAS": ExtrasSettings,
        }

        self._training_additional_required = {
            "TRAINING": TrainingConfig,
        }

        self._training_additional_optional = {
            "MLFLOW": MLflowSettings,
            "OPTUNA": OptunaSettings,
        }

    def load_training_settings(self, config_path: Path | str) -> TrainingWorkflowSettings:
        """Load settings optimized for training workflows with lazy validation.

        **Lazy Validation**: Settings are loaded without Pydantic validation by default.
        Validation happens at build time when components are constructed, not at load time.
        This allows incomplete configs with placeholder values or missing optional data.

        Only loads sections required for training:
        - Required: SESSION, DATAMODULE, DATASET, TRAINING
        - Optional: MODEL, MLFLOW, OPTUNA, PATHS, EXTRAS

        Args:
            config_path: Path to TOML configuration file

        Returns:
            TrainingWorkflowSettings: Training-specific settings instance (unvalidated)

        Raises:
            FileNotFoundError: If config file doesn't exist
            ConfigSectionError: If required sections are missing from TOML file

        Note:
            Validation errors (e.g., invalid paths, missing fields) will only surface
            when settings are used to build components. Use `settings.model_validate()`
            to validate explicitly before build if early error detection is needed.
        """
        from dlkit.tools.io.config import load_sections_config, check_section_exists

        # Combine required sections for training
        required_sections = {
            **self._base_required_sections,
            **self._training_additional_required,
        }

        # Combine optional sections
        optional_sections = {
            **self._base_optional_sections,
            **self._training_additional_optional,
        }

        # Load required sections
        sections_data = load_sections_config(config_path, required_sections)

        # Load optional sections that exist
        for section_name, model_class in optional_sections.items():
            if check_section_exists(config_path, section_name):
                optional_data = load_sections_config(config_path, {section_name: model_class})
                sections_data.update(optional_data)

        # Create training settings instance
        # Cast sections_data to proper types for constructor
        typed_kwargs: dict[str, Any] = {}
        for key, value in sections_data.items():
            if key == "SESSION":
                typed_kwargs[key] = cast(SessionSettings, value)
            elif key == "DATAMODULE":
                typed_kwargs[key] = cast(DataModuleSettings, value)
            elif key == "DATASET":
                typed_kwargs[key] = cast(DatasetSettings, value)
            elif key == "TRAINING":
                typed_kwargs[key] = cast(TrainingConfig, value)
            elif key == "MODEL":
                typed_kwargs[key] = cast(ModelComponentSettings, value)
            elif key == "MLFLOW":
                typed_kwargs[key] = cast(MLflowSettings, value)
            elif key == "OPTUNA":
                typed_kwargs[key] = cast(OptunaSettings, value)
            elif key == "PATHS":
                typed_kwargs[key] = cast(PathsSettings, value)
            elif key == "EXTRAS":
                typed_kwargs[key] = cast(ExtrasSettings, value)

        return TrainingWorkflowSettings(**typed_kwargs)



    def create_settings_for_workflow(
        self,
        config_path: Path | str,
        workflow_type: str
    ) -> BaseWorkflowSettings:
        """Factory method to create settings based on workflow type.

        This method implements the Factory Method pattern, allowing
        for easy extension with new workflow types following OCP.

        Args:
            config_path: Path to TOML configuration file
            workflow_type: Type of workflow ("training", "inference", "general")

        Returns:
            BaseWorkflowSettings: Appropriate settings instance

        Raises:
            ValueError: If workflow_type is not supported
        """
        workflow_factories = {
            "training": self.load_training_settings,
            # "inference": Removed - use inference API instead
        }

        if workflow_type not in workflow_factories:
            available_types = list(workflow_factories.keys())
            raise ValueError(
                f"Unsupported workflow type: {workflow_type}. "
                f"Available types: {available_types}"
            )

        factory_method = workflow_factories[workflow_type]
        return factory_method(config_path)

    def load_sections(
        self,
        config_path: Path | str,
        sections: list[str],
        *,
        strict: bool = False
    ) -> BaseWorkflowSettings:
        """Load arbitrary combination of configuration sections for maximum flexibility.

        This method gives users complete control over which sections to load,
        enabling any workflow combination. Sections can be treated as optional
        (default) or strict (must exist) based on the strict parameter.

        Args:
            config_path: Path to TOML configuration file
            sections: List of section names to load (e.g., ['MODEL', 'DATASET', 'MLFLOW'])
            strict: If True, all specified sections must exist; if False (default),
                   missing sections are ignored for maximum flexibility

        Returns:
            BaseWorkflowSettings: Settings containing only the requested sections

        Raises:
            ValueError: If no valid sections are specified or unknown sections are requested
            FileNotFoundError: If config file doesn't exist
            ConfigSectionError: If strict=True and required sections are missing
            ConfigValidationError: If validation fails

        Examples:
            >>> # Custom evaluation workflow
            >>> settings = loader.load_sections("config.toml", ["MODEL", "DATASET"])
            >>>
            >>> # Experiment tracking only
            >>> settings = loader.load_sections("config.toml", ["MLFLOW", "OPTUNA"])
            >>>
            >>> # Data pipeline configuration
            >>> settings = loader.load_sections("config.toml", ["DATAMODULE", "DATASET", "PATHS"])
            >>>
            >>> # Complete training setup (user-controlled)
            >>> settings = loader.load_sections("config.toml", [
            ...     "SESSION", "MODEL", "DATAMODULE", "DATASET", "TRAINING", "MLFLOW"
            ... ])
            >>>
            >>> # Strict loading (all sections must exist)
            >>> settings = loader.load_sections("config.toml", ["MODEL", "DATASET"], strict=True)
        """
        from dlkit.tools.io.config import load_sections_config, check_section_exists

        if not sections:
            raise ValueError("At least one section must be specified")

        # Map section names to model classes
        section_model_map = {
            "SESSION": SessionSettings,
            "MODEL": ModelComponentSettings,
            "DATAMODULE": DataModuleSettings,
            "DATASET": DatasetSettings,
            "TRAINING": TrainingConfig,
            "MLFLOW": MLflowSettings,
            "OPTUNA": OptunaSettings,
            "PATHS": PathsSettings,
            "EXTRAS": ExtrasSettings,
        }

        # Validate requested sections
        unknown_sections = [s for s in sections if s not in section_model_map]
        if unknown_sections:
            available_sections = list(section_model_map.keys())
            raise ValueError(
                f"Unknown sections: {unknown_sections}. "
                f"Available sections: {available_sections}"
            )

        # Build loading plan based on strict mode
        sections_to_load = {}
        missing_sections = []

        for section_name in sections:
            section_exists = check_section_exists(config_path, section_name)

            if section_exists:
                sections_to_load[section_name] = section_model_map[section_name]
            elif strict:
                missing_sections.append(section_name)
            # In non-strict mode, missing sections are silently ignored

        # Handle strict mode violations
        if strict and missing_sections:
            raise ValueError(
                f"Strict mode: Required sections missing from config file: {missing_sections}"
            )

        # Load existing sections
        if sections_to_load:
            sections_data = load_sections_config(config_path, sections_to_load)
        else:
            sections_data = {}

        # Determine the best settings class based on requested sections
        has_training = "TRAINING" in sections_data
        has_mlflow = "MLFLOW" in sections_data
        has_optuna = "OPTUNA" in sections_data

        # Cast all loaded sections to proper types
        typed_kwargs: dict[str, Any] = {}

        for section_name in sections_data:
            value = sections_data[section_name]
            if section_name == "SESSION":
                typed_kwargs[section_name] = cast(SessionSettings, value)
            elif section_name == "DATAMODULE":
                typed_kwargs[section_name] = cast(DataModuleSettings, value)
            elif section_name == "DATASET":
                typed_kwargs[section_name] = cast(DatasetSettings, value)
            elif section_name == "TRAINING":
                typed_kwargs[section_name] = cast(TrainingConfig, value)
            elif section_name == "MODEL":
                typed_kwargs[section_name] = cast(ModelComponentSettings, value)
            elif section_name == "MLFLOW":
                typed_kwargs[section_name] = cast(MLflowSettings, value)
            elif section_name == "OPTUNA":
                typed_kwargs[section_name] = cast(OptunaSettings, value)
            elif section_name == "PATHS":
                typed_kwargs[section_name] = cast(PathsSettings, value)
            elif section_name == "EXTRAS":
                typed_kwargs[section_name] = cast(ExtrasSettings, value)

        # Choose the appropriate settings class based on loaded sections
        if has_training and (has_mlflow or has_optuna):
            # Full training workflow
            return TrainingWorkflowSettings(**typed_kwargs)
        elif has_training:
            # Basic training workflow
            return TrainingWorkflowSettings(**typed_kwargs)
        elif "SESSION" in typed_kwargs and hasattr(typed_kwargs["SESSION"], "inference") and typed_kwargs["SESSION"].inference:
            # Inference workflow
            return InferenceWorkflowSettings(**typed_kwargs)
        else:
            # Base workflow for any other combination
            return BaseWorkflowSettings(**typed_kwargs)

# Default factory instance for convenience
default_settings_loader = PartialSettingsLoader()

# Convenience functions that delegate to the default loader

def load_settings(
    config_path: Path | str,
    *,
    inference: bool = False,
    sections: list[str] | None = None,
    strict: bool = False
) -> BaseWorkflowSettings:
    """Primary configuration loading API with flexible loading strategies.

    **Standard Pattern**: For training workflows, use `load_training_settings()`
    directly instead of this function. This function is for advanced use cases
    requiring custom section combinations.

    Loading Strategies (in order of preference):
    1. **Training workflows**: Use `load_training_settings()` (recommended)
    2. **Custom sections**: Use `sections=["MODEL", "DATASET"]` for partial loading
    3. **Inference**: Use `dlkit.interfaces.inference.infer()` API instead

    Args:
        config_path: Path to TOML configuration file
        inference: Deprecated - use `dlkit.interfaces.inference.infer()` instead
        sections: List of specific sections to load (for custom workflows)
        strict: If True with sections, all specified sections must exist

    Returns:
        BaseWorkflowSettings: Settings instance with requested sections

    Examples:
        >>> # RECOMMENDED: Use specific functions for common workflows
        >>> from dlkit.tools.config import load_training_settings
        >>> settings = load_training_settings("config.toml")
        >>>
        >>> # ADVANCED: Custom section combinations
        >>> settings = load_settings("config.toml", sections=["MODEL", "DATASET"])
        >>>
        >>> # DEPRECATED: Inference mode (use inference API instead)
        >>> # settings = load_settings("config.toml", inference=True)  # DON'T DO THIS
        >>> from dlkit.interfaces.inference import infer
        >>> result = infer(checkpoint_path, inputs)  # DO THIS INSTEAD

    Raises:
        ValueError: If inference=True (deprecated, use inference API)
        ConfigSectionError: If requested sections are missing
    """
    if sections is not None:
        # Strategy: Custom section loading (advanced use cases)
        return default_settings_loader.load_sections(config_path, sections, strict=strict)
    elif inference:
        # Inference mode is removed - direct users to proper API
        raise ValueError(
            "Inference mode has been removed from config loading. "
            "Use the inference API instead: dlkit.interfaces.inference.infer()"
        )
    else:
        # Default: Training mode
        return default_settings_loader.load_training_settings(config_path)


def load_training_settings(config_path: Path | str) -> TrainingWorkflowSettings:
    """Load configuration for training workflows (RECOMMENDED).

    **This is the standard entry point for loading training configurations.**

    Loads sections optimized for training workflows with lazy validation:
    - Required: SESSION, DATAMODULE, DATASET, TRAINING
    - Optional: MODEL, MLFLOW, OPTUNA, PATHS, EXTRAS

    Lazy Validation Behavior:
    - Fields are loaded WITHOUT Pydantic defaults (partial configs supported)
    - Type coercion is applied (str→Path, dict→NestedModel, etc.)
    - Validation happens at build time, not load time
    - Enables programmatic overrides via `update_settings()`

    Args:
        config_path: Path to TOML configuration file

    Returns:
        TrainingWorkflowSettings: Training settings with lazy validation

    Examples:
        >>> # Standard training workflow
        >>> from dlkit.tools.config import load_training_settings
        >>> settings = load_training_settings("config.toml")
        >>>
        >>> # Partial config with programmatic overrides
        >>> settings = load_training_settings("partial_config.toml")
        >>> settings = update_settings(settings, {
        ...     "DATASET": {"features": [...], "targets": [...]}
        ... })
        >>>
        >>> # Validate explicitly if needed
        >>> validated = settings.model_validate(settings.model_dump())

    See Also:
        - `update_settings()`: Merge updates into settings
        - `load_sections()`: Load custom section combinations (advanced)
    """
    return default_settings_loader.load_training_settings(config_path)


# BREAKING CHANGE: load_inference_settings removed
# Use inference API instead:
# from dlkit.interfaces.inference import infer
# result = infer(checkpoint_path, inputs)




def load_sections(
    config_path: Path | str,
    sections: list[str],
    *,
    strict: bool = False
) -> BaseWorkflowSettings:
    """Load arbitrary configuration sections with maximum flexibility.

    **Advanced Use Only**: For most workflows, use `load_settings()` or
    `load_training_settings()` instead. This function is for custom workflows
    that need specific section combinations.

    Args:
        config_path: Path to TOML configuration file
        sections: List of section names to load
        strict: If True, all specified sections must exist; if False (default),
               missing sections are ignored for maximum flexibility

    Returns:
        BaseWorkflowSettings: Settings with only requested sections

    Examples:
        >>> # Load only model and dataset configuration (for evaluation)
        >>> settings = load_sections("config.toml", ["MODEL", "DATASET"])
        >>>
        >>> # Load experiment tracking configuration only
        >>> settings = load_sections("config.toml", ["MLFLOW", "OPTUNA"])
        >>>
        >>> # Strict loading ensures all sections exist
        >>> settings = load_sections("config.toml", ["MODEL", "DATASET"], strict=True)

    Note:
        For standard training workflows, prefer `load_training_settings()`.
        For partial configs with programmatic overrides, use `load_settings()`.
    """
    return default_settings_loader.load_sections(config_path, sections, strict=strict)