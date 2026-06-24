"""Settings factory for efficient partial loading following SOLID principles.

Note: WorkflowSettingsLoader and load_settings() are legacy entrypoints that
dispatch on SESSION.workflow. They will be replaced by load_job() in Task 2
of the config protocol redesign. The JobConfig hierarchy is now the primary API.
"""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from dlkit.common.errors import ConfigValidationError
from dlkit.infrastructure.config.validators import (
    ConfigValidationError as _ValidatorConfigValidationError,
)

if TYPE_CHECKING:
    from dlkit.infrastructure.config.inference_workflow_settings import (
        InferenceWorkflowSettings,
    )
    from dlkit.infrastructure.config.job_config import (
        InferenceJobConfig,
        SearchJobConfig,
        TrainingJobConfig,
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
            raise _ValidatorConfigValidationError(
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
        if env := _read_env_patches("DLKIT_", "__", uppercase_section=True):
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


# ---------------------------------------------------------------------------
# New JobConfig loader (Task 2)
# ---------------------------------------------------------------------------

_PROFILE_KEYS: tuple[str, ...] = ("model", "data", "training", "tracking")

type _TomlScalar = str | int | float | bool | None
type _TomlValue = _TomlScalar | list["_TomlValue"] | dict[str, "_TomlValue"]
type _TomlDict = dict[str, _TomlValue]


def _deep_merge(base: _TomlDict, override: _TomlDict) -> _TomlDict:
    """Recursive key-by-key merge; override wins at leaf values.

    Args:
        base: Base dictionary (lower priority).
        override: Override dictionary (higher priority).

    Returns:
        New merged dictionary. Neither input is mutated.
    """
    result: _TomlDict = dict(base)
    for key, val in override.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(val, dict):
            result[key] = _deep_merge(existing, val)
        else:
            result[key] = val
    return result


def load_job(
    config_paths: Path | str | Sequence[Path | str],
    run_type: str | None = None,
) -> TrainingJobConfig | InferenceJobConfig | SearchJobConfig:
    """Load and validate a job config from one or more TOML files.

    Multiple paths are merged left-to-right (later files win). Profile references
    in ``[run]`` (keys ``model``, ``data``, ``training``, ``tracking``) are resolved
    before Pydantic validation: the referenced TOML file is loaded and its section
    content is merged as the base for the corresponding top-level section.

    Args:
        config_paths: One or more TOML paths merged left-to-right (later wins).
        run_type: Override ``run.type``. Required when ``run.type`` is absent from TOML.

    Returns:
        A validated typed job config matching the resolved ``run.type``.

    Raises:
        ConfigValidationError: Missing ``run.type``, bad profile section, or
            Pydantic validation failure.
        FileNotFoundError: If a config file or referenced profile does not exist.
    """
    from dlkit.infrastructure.config.core.sources import DLKitTomlSource, _read_env_patches
    from dlkit.infrastructure.config.job_config import (
        InferenceJobConfig,
        SearchJobConfig,
        TrainingJobConfig,
    )

    paths = (
        [Path(config_paths)]
        if isinstance(config_paths, (str, Path))
        else [Path(p) for p in config_paths]
    )

    # 1. Merge all job files left-to-right (later wins).
    merged: _TomlDict = {}
    for path in paths:
        raw = DLKitTomlSource(path)()
        merged = _deep_merge(merged, raw)

    # 2. Resolve typed profile references from run.*.
    run_raw = merged.get("run", {})
    job_dir = paths[0].parent
    profile_base: _TomlDict = {}

    if isinstance(run_raw, dict):
        for section_key in _PROFILE_KEYS:
            profile_path_str = run_raw.get(section_key)
            if not isinstance(profile_path_str, str):
                continue  # absent or already a dict — not a profile reference
            profile_path = (job_dir / profile_path_str).resolve()
            profile_raw = DLKitTomlSource(profile_path)()
            if section_key not in profile_raw:
                raise ConfigValidationError(
                    f"Profile '{profile_path_str}' referenced as run.{section_key} "
                    f"must contain a [{section_key}] section. "
                    f"Found sections: {list(profile_raw.keys())}."
                )
            profile_base[section_key] = profile_raw[section_key]

    # 3. Merge profiles as base; job-file sections win.
    merged = _deep_merge(profile_base, merged)

    # 4. Remove profile path strings from run (they were references, not config).
    run_section = merged.get("run")
    if isinstance(run_section, dict):
        cleaned_run = {
            k: v for k, v in run_section.items() if not (k in _PROFILE_KEYS and isinstance(v, str))
        }
        merged = {**merged, "run": cleaned_run}

    # 5. Resolve run.type.
    effective_run_raw = merged.get("run", {})
    effective_run: _TomlDict = effective_run_raw if isinstance(effective_run_raw, dict) else {}
    toml_type = effective_run.get("type")
    resolved_type: str | None = run_type or (toml_type if isinstance(toml_type, str) else None)
    if resolved_type is None:
        raise ConfigValidationError(
            'No run.type found. Set [run] type = "train" in the TOML '
            "or pass run_type= to load_job()."
        )
    merged = {**merged, "run": {**effective_run, "type": resolved_type}}

    # 6. Apply DLKIT_* env patches.
    patches = _read_env_patches("DLKIT")
    if patches:
        merged = _deep_merge(merged, patches)

    # 7. Dispatch to typed subtype.
    try:
        match resolved_type:
            case "train":
                return TrainingJobConfig.model_validate(merged)
            case "predict":
                return InferenceJobConfig.model_validate(merged)
            case "search":
                return SearchJobConfig.model_validate(merged)
            case _:
                raise ConfigValidationError(
                    f"Unknown run.type: {resolved_type!r}. Must be 'train', 'predict', or 'search'."
                )
    except ConfigValidationError:
        raise
    except Exception as exc:
        raise ConfigValidationError(
            f"Config validation failed for run.type={resolved_type!r}: {exc}"
        ) from exc
