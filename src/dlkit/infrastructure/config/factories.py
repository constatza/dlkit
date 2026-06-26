"""Settings factory — JobConfig-based TOML loader."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dlkit.common.errors import ConfigValidationError

if TYPE_CHECKING:
    from dlkit.infrastructure.config.job_config import (
        InferenceJobConfig,
        SearchJobConfig,
        TrainingJobConfig,
    )

# ---------------------------------------------------------------------------
# JobConfig loader
# ---------------------------------------------------------------------------

_PROFILE_KEYS: tuple[str, ...] = ("model", "data", "training", "tracking")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive key-by-key merge; override wins at leaf values.

    Args:
        base: Base dictionary (lower priority).
        override: Override dictionary (higher priority).

    Returns:
        New merged dictionary. Neither input is mutated.
    """
    result: dict[str, Any] = dict(base)
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
    merged: dict[str, Any] = {}
    for path in paths:
        raw = DLKitTomlSource(path)()
        merged = _deep_merge(merged, raw)

    # 2. Resolve typed profile references from run.*.
    run_raw = merged.get("run", {})
    job_dir = paths[0].parent
    profile_base: dict[str, Any] = {}

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
    effective_run: dict[str, Any] = effective_run_raw if isinstance(effective_run_raw, dict) else {}
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
