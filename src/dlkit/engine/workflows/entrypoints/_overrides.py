"""Runtime-owned helpers for request-level overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dlkit.infrastructure.config.core.base_settings import BasicSettings
from dlkit.infrastructure.io import locations
from dlkit.infrastructure.io.paths import normalize_user_path

_PATH_FIELDS = frozenset({"checkpoint_path", "root_dir", "output_dir", "data_dir"})


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
                require_absolute=(key == "root_dir"),
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


def _build_patch(settings: Any, overrides: dict[str, Any]) -> dict[str, Any]:
    """Build a validated settings patch from request overrides."""
    patch: dict[str, Any] = {}

    if (cp := overrides.get("checkpoint_path")) and settings.MODEL:
        patch["MODEL.checkpoint"] = cp

    if settings.TRAINING:
        if (epochs := overrides.get("epochs")) is not None:
            patch["TRAINING.epochs"] = epochs
            patch["TRAINING.trainer.max_epochs"] = epochs
        if (lr := overrides.get("learning_rate")) is not None:
            patch["TRAINING.optimizer.lr"] = float(lr)
        if loss := overrides.get("loss_function"):
            patch["TRAINING.loss_function"] = {
                "name": loss,
                "module_path": overrides.get("loss_module", "dlkit.domain.losses"),
            }

    if (bs := overrides.get("batch_size")) is not None and settings.DATAMODULE:
        patch["DATAMODULE.dataloader.batch_size"] = bs

    mlflow_fields = {
        key: value
        for key, value in overrides.items()
        if key in ("experiment_name", "run_name", "register_model", "tags") and value is not None
    }
    if mlflow_fields:
        patch["MLFLOW"] = mlflow_fields

    optuna_fields = {
        key: value
        for key, value in {
            "enabled": overrides.get("enable_optuna"),
            "n_trials": overrides.get("trials"),
            "study_name": overrides.get("study_name"),
        }.items()
        if value is not None
    }
    match (bool(optuna_fields), bool(settings.OPTUNA)):
        case (False, _):
            pass
        case (True, True):
            patch["OPTUNA"] = optuna_fields
        case (True, False) if optuna_fields.get("enabled"):
            patch["OPTUNA"] = {
                "enabled": True,
                "n_trials": optuna_fields.get("n_trials", 3),
                "study_name": optuna_fields.get("study_name", "default_study"),
                "storage": locations.optuna_storage_uri(),
            }

    return patch


def apply_runtime_overrides[T: BasicSettings](
    base_settings: T,
    **overrides: Any,
) -> T:
    """Apply runtime overrides onto immutable settings models (pure function)."""
    patch = _build_patch(base_settings, overrides)
    return base_settings.patch(patch) if patch else base_settings
