"""Settings factory for efficient partial loading following SOLID principles."""

from __future__ import annotations

from pathlib import Path

from .workflow_settings import (
    BaseWorkflowSettings,
    TrainingWorkflowSettings,
)


class WorkflowSettingsLoader:
    """Factory class for config loading.

    Thin facade over :meth:`TrainingWorkflowSettings.from_toml` that preserves
    the public ``load_settings`` / ``load_sections`` API surface.
    """

    def load_training_settings(self, config_path: Path | str) -> TrainingWorkflowSettings:
        """Load full training configuration from a TOML file.

        Args:
            config_path: Path to TOML configuration file.

        Returns:
            TrainingWorkflowSettings: Training-specific settings instance.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            pydantic.ValidationError: If validation fails.
        """
        return TrainingWorkflowSettings.from_toml(config_path)

    def create_settings_for_workflow(
        self, config_path: Path | str, workflow_type: str
    ) -> BaseWorkflowSettings:
        """Factory method to create settings based on workflow type.

        Args:
            config_path: Path to TOML configuration file.
            workflow_type: Type of workflow. Only ``"training"`` is supported.

        Returns:
            BaseWorkflowSettings: Appropriate settings instance.

        Raises:
            ValueError: If workflow_type is not supported.
        """
        if workflow_type != "training":
            raise ValueError(
                f"Unsupported workflow type: {workflow_type}. Available types: ['training']"
            )
        return self.load_training_settings(config_path)

    def load_sections(
        self, config_path: Path | str, sections: list[str], *, strict: bool = False
    ) -> BaseWorkflowSettings:
        """Load an arbitrary combination of configuration sections.

        Args:
            config_path: Path to TOML configuration file.
            sections: List of section names to load.
            strict: If ``True``, all specified sections must exist in the file.

        Returns:
            BaseWorkflowSettings: Settings containing only the requested sections.

        Raises:
            ValueError: If no sections are specified or strict mode detects missing sections.
            FileNotFoundError: If config file doesn't exist.
        """
        if not sections:
            raise ValueError("At least one section must be specified")

        if strict:
            from dlkit.tools.io.config import get_available_sections
            available = get_available_sections(config_path)
            missing = [s for s in sections if s not in available]
            if missing:
                raise ValueError(
                    f"Strict mode: Required sections missing from config file: {missing}"
                )

        return TrainingWorkflowSettings.from_toml(
            config_path, sections=[s.upper() for s in sections]
        )


# Default factory instance for convenience
default_settings_loader = WorkflowSettingsLoader()

# Convenience functions that delegate to the default loader


def load_settings(config_path: Path | str) -> TrainingWorkflowSettings:
    """Load full training configuration from TOML file.

    This is the primary API for loading DLKit configurations. It loads all
    standard sections for training workflows with eager validation.

    **Loaded sections:**
    - Required: SESSION, TRAINING
    - Optional: DATAMODULE, DATASET, MODEL, MLFLOW, OPTUNA, PATHS, EXTRAS

    Args:
        config_path: Path to TOML configuration file.

    Returns:
        TrainingWorkflowSettings: Complete training settings.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ConfigSectionError: If required sections are missing.

    Examples:
        >>> # Load from tools.io (preferred)
        >>> from dlkit.tools.io import load_settings
        >>> settings = load_settings("config.toml")
        >>>
        >>> # Also available from tools.config
        >>> from dlkit.tools.config import load_settings
        >>> settings = load_settings("config.toml")

    See Also:
        - `load_sections()`: Load specific sections for custom workflows.
        - `load_model()`: For inference workflows.
    """
    return default_settings_loader.load_training_settings(config_path)


# BREAKING CHANGE: load_inference_settings removed
# Use inference API instead:
# from dlkit.interfaces.inference import infer
# result = infer(checkpoint_path, inputs)


def load_sections(
    config_path: Path | str, sections: list[str], *, strict: bool = False
) -> BaseWorkflowSettings:
    """Load specific configuration sections for custom workflows.

    For most workflows, use `load_settings()` instead. This function is for
    advanced use cases requiring specific section combinations.

    Args:
        config_path: Path to TOML configuration file.
        sections: List of section names to load (e.g., ["MODEL", "DATASET"]).
        strict: If True, all specified sections must exist; if False (default),
               missing sections are ignored.

    Returns:
        BaseWorkflowSettings: Settings with only the requested sections.

    Raises:
        ValueError: If no valid sections specified or unknown sections requested.
        FileNotFoundError: If config file doesn't exist.
        ConfigSectionError: If strict=True and required sections are missing.

    Examples:
        >>> # Load from tools.io (preferred)
        >>> from dlkit.tools.io import load_sections
        >>> settings = load_sections("config.toml", ["MODEL", "DATASET"])
        >>>
        >>> # Strict loading ensures all sections exist
        >>> settings = load_sections("config.toml", ["MODEL", "DATASET"], strict=True)

    See Also:
        - `load_settings()`: Load full training configuration (recommended).
    """
    return default_settings_loader.load_sections(config_path, sections, strict=strict)


# Backward-compat alias
PartialSettingsLoader = WorkflowSettingsLoader
