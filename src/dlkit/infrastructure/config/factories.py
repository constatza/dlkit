"""Settings factory for efficient partial loading following SOLID principles.

Note: WorkflowSettingsLoader and load_settings() are legacy entrypoints that
dispatch on SESSION.workflow. They will be replaced by load_job() in Task 2
of the config protocol redesign. The JobConfig hierarchy is now the primary API.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

from dlkit.infrastructure.config.validators import ConfigValidationError

if TYPE_CHECKING:
    from dlkit.infrastructure.config.inference_workflow_settings import (
        InferenceWorkflowSettings,
    )
    from dlkit.infrastructure.config.training_workflow_settings import (
        TrainingWorkflowSettings,
    )

# Union of all concrete types returned by load_settings() / load_sections().
# Extended in Task 3 when OptimizationWorkflowSettings is introduced.
WorkflowSettings = "TrainingWorkflowSettings | InferenceWorkflowSettings"


class WorkflowSettingsLoader:
    """Legacy factory class for config loading.

    Dispatches on SESSION.workflow to return the appropriate workflow config type.

    Note: Deprecated in favour of ``load_job()`` (Task 2). Kept for backward
    compatibility until engine and CLI are wired to ``JobConfig``.
    """

    def load_settings(
        self, config_path: Path | str
    ) -> TrainingWorkflowSettings | InferenceWorkflowSettings:
        """Load workflow config from TOML, dispatching on SESSION.workflow.

        Args:
            config_path: Path to TOML configuration file.

        Returns:
            The appropriate workflow config type based on SESSION.workflow.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValueError: If SESSION.workflow is unknown.
            pydantic.ValidationError: If validation fails.
        """
        from dlkit.infrastructure.config.core.patching import patch_model
        from dlkit.infrastructure.config.core.sources import DLKitTomlSource, _read_env_patches
        from dlkit.infrastructure.config.workflow_settings import (
            InferenceWorkflowSettings,
            TrainingWorkflowSettings,
        )

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
            case "train" | "optimize":
                config = TrainingWorkflowSettings.model_validate(toml_data)
            case "inference":
                config = InferenceWorkflowSettings.model_validate(toml_data)
            case _:
                raise ValueError(
                    f"Unknown SESSION.workflow value: {mode!r}. "
                    "Expected 'train', 'optimize', or 'inference'."
                )

        # Apply environment variable patches if present
        if env := _read_env_patches("DLKIT_", "__"):
            config = patch_model(config, env)

        return config


# Default factory instance for convenience
default_settings_loader = WorkflowSettingsLoader()


# Convenience function that delegates to the default loader


def load_settings(
    config_path: Path | str,
) -> TrainingWorkflowSettings | InferenceWorkflowSettings:
    """Load workflow configuration from TOML file with automatic dispatching.

    This is a legacy entrypoint. Use ``load_job()`` (Task 2) for new code.

    Args:
        config_path: Path to TOML configuration file.

    Returns:
        Workflow settings object based on SESSION.workflow.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If SESSION.workflow is unknown or validation fails.
    """
    return default_settings_loader.load_settings(config_path)


def load_sections(
    config_path: Path | str, sections: list[str], *, strict: bool = False
) -> TrainingWorkflowSettings | InferenceWorkflowSettings:
    """Load specific configuration sections for custom workflows.

    For most workflows, use ``load_settings()`` instead. This function is for
    advanced use cases requiring specific section combinations. It still dispatches
    on SESSION.workflow to return the appropriate workflow config type.

    Args:
        config_path: Path to TOML configuration file.
        sections: List of section names to load (e.g., ["MODEL", "DATASET"]).
        strict: If True, all specified sections must exist; if False (default),
               missing sections are ignored.

    Returns:
        Workflow settings object based on SESSION.workflow.

    Raises:
        ValueError: If no valid sections specified or unknown sections requested.
        FileNotFoundError: If config file doesn't exist.
    """
    if not sections:
        raise ValueError("At least one section must be specified")

    path = Path(config_path)

    if strict:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
        available = [key.upper() for key in data if isinstance(key, str)]
        missing = [s for s in sections if s not in available]
        if missing:
            raise ValueError(f"Strict mode: Required sections missing from config file: {missing}")

    # Still dispatch based on SESSION.workflow
    return default_settings_loader.load_settings(path)
