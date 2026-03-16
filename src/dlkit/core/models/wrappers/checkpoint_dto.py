"""Typed checkpoint data transfer objects and normalization helpers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ModelCheckpointDTO(BaseModel):
    """Flat DTO for model settings stored in a checkpoint.

    Replaces the nested ``{"params": {...}}`` structure with a flat contract
    shared by both writer (on_save_checkpoint) and reader (build_model_from_checkpoint).

    Attributes:
        name: Model class name or type path.
        module_path: Python module path for importing the model class.
        resolved_init_kwargs: Exact kwargs passed to the model constructor.
            Stored at training time so inference can reproduce the same call
            without guessing or re-running shape inference.
        all_hyperparams: Full settings snapshot for reference / debugging.
    """

    name: str
    module_path: str
    resolved_init_kwargs: dict[str, Any] = Field(default_factory=dict)
    all_hyperparams: dict[str, Any] = Field(default_factory=dict)


def _normalize_model_settings(model_settings_raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy model_settings dicts to the flat DTO shape.

    Legacy shape::

        {
            "name": str,
            "module_path": str,
            "params": {hyperparams...},
            "class_name": str,
        }

    Normalized shape::

        {
            "name": str,
            "module_path": str,
            "resolved_init_kwargs": {hyperparams...},
            "all_hyperparams": {hyperparams...},
        }

    Args:
        model_settings_raw: Raw model settings dict from checkpoint metadata.

    Returns:
        Dict compatible with ``ModelCheckpointDTO``.
    """
    if "resolved_init_kwargs" in model_settings_raw and "all_hyperparams" in model_settings_raw:
        return model_settings_raw

    params = model_settings_raw.get("params") or {}
    name = model_settings_raw.get("name") or ""
    module_path = model_settings_raw.get("module_path") or ""
    flat_kwargs = {
        key: value
        for key, value in model_settings_raw.items()
        if key not in {"name", "module_path", "params", "class_name"}
    }
    resolved_init_kwargs = {**params, **flat_kwargs}
    return {
        "name": name,
        "module_path": module_path,
        "resolved_init_kwargs": resolved_init_kwargs,
        "all_hyperparams": resolved_init_kwargs,
    }


def normalize_checkpoint_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Normalize checkpoint metadata without keeping an explicit format tag.

    Args:
        metadata: ``checkpoint["dlkit_metadata"]`` dict.

    Returns:
        Shallow copy with normalized ``model_settings`` and no ``version`` entry.
    """
    migrated = dict(metadata)

    if "model_settings" in migrated and isinstance(migrated["model_settings"], dict):
        migrated["model_settings"] = _normalize_model_settings(migrated["model_settings"])

    migrated.pop("version", None)
    return migrated
