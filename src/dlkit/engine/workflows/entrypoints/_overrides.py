"""Runtime-owned helpers for request-level overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dlkit.infrastructure.config.core.base_settings import BasicSettings
from dlkit.infrastructure.io.paths import normalize_user_path

_PATH_FIELDS = frozenset({"checkpoint_path"})


def normalize_override_path(
    path: str | Path | None,
    *,
    require_absolute: bool = False,
) -> Path | None:
    """Normalize path overrides while preserving relative paths when requested."""
    normalized = normalize_user_path(path, require_absolute=require_absolute)
    if require_absolute:
        return normalized

    if path is None:
        return None

    if isinstance(path, Path):
        if path.is_absolute() or "~" in str(path):
            return normalized
        return path

    if isinstance(path, str):
        if "~" in path or Path(path).is_absolute():
            return normalized
        return Path(path)

    return normalized


def build_runtime_overrides(**kwargs: Any) -> dict[str, Any]:
    """Build a flattened override dictionary with path normalization."""
    additional = kwargs.pop("additional_overrides", None) or {}

    normalized = {}
    for key, value in kwargs.items():
        if key in _PATH_FIELDS:
            normalized[key] = normalize_override_path(
                value,
                require_absolute=False,
            )
        else:
            normalized[key] = value

    result = {key: value for key, value in normalized.items() if value is not None}
    if additional:
        result.update({key: value for key, value in additional.items() if value is not None})
    return result


def validate_runtime_overrides(**overrides: Any) -> list[str]:
    """Validate request-level overrides that are not model-validated."""
    errors: list[str] = []

    if "checkpoint_path" in overrides and overrides["checkpoint_path"] is not None:
        checkpoint_path = overrides["checkpoint_path"]
        if not isinstance(checkpoint_path, Path):
            try:
                checkpoint_path = Path(checkpoint_path)
            except TypeError, ValueError:
                errors.append("checkpoint_path must be a valid path")
                return errors

        if not checkpoint_path.exists():
            errors.append(f"Checkpoint file does not exist: {checkpoint_path}")

    return errors


def _build_patch(settings: BasicSettings, overrides: dict[str, Any]) -> dict[str, Any]:
    """Build a settings patch from flat request overrides onto a JobConfig.

    Args:
        settings: A JobConfig instance to patch.
        overrides: Flat mapping of override keys to values.

    Returns:
        Nested patch dict suitable for ``settings.patch()``.
    """
    patch: dict[str, Any] = {}

    if cp := overrides.get("checkpoint_path"):
        patch["model"] = {"checkpoint": cp}

    training: dict[str, Any] = {}
    if (epochs := overrides.get("epochs")) is not None:
        training.setdefault("trainer", {})["max_epochs"] = epochs
    if (lr := overrides.get("learning_rate")) is not None:
        training.setdefault("optimizer", {}).setdefault("default_optimizer", {})["lr"] = float(lr)
    if loss := overrides.get("loss_function"):
        training["loss"] = {
            "name": loss,
            "module_path": overrides.get("loss_module", "dlkit.domain.losses"),
        }
    if training:
        patch["training"] = training

    if (bs := overrides.get("batch_size")) is not None:
        patch["data"] = {"batch_size": bs}

    experiment: dict[str, Any] = {}
    if name := overrides.get("experiment_name"):
        experiment["name"] = name
    if run_name := overrides.get("run_name"):
        experiment["run_name"] = run_name
    if (register := overrides.get("register_model")) is not None:
        experiment["register_model"] = register
    if tags := overrides.get("tags"):
        experiment["tags"] = tags
    if experiment:
        patch["experiment"] = experiment

    search: dict[str, Any] = {}
    if trials := overrides.get("trials"):
        search["n_trials"] = trials
    if study_name := overrides.get("study_name"):
        search["study_name"] = study_name
    if search:
        patch["search"] = search

    return patch


def apply_runtime_overrides[T: BasicSettings](
    base_settings: T,
    **overrides: Any,
) -> T:
    """Apply runtime overrides onto immutable settings models (pure function)."""
    patch = _build_patch(base_settings, overrides)
    return base_settings.patch(patch) if patch else base_settings
