"""Checkpoint reading and inspection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from dlkit.shared.errors import WorkflowError
from dlkit.tools.config.model_components import ModelComponentSettings
from dlkit.tools.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointValidationResult:
    """Result of checkpoint validation with typed metadata."""

    checkpoint_path: Path
    exists: bool
    valid_format: bool
    has_state_dict: bool
    has_model_settings: bool
    has_shape_metadata: bool
    dtype: torch.dtype | None


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointInfo:
    """Structured checkpoint metadata without full model loading."""

    checkpoint_path: Path
    has_dlkit_metadata: bool
    has_hyper_parameters: bool
    model_family: str | None = None
    wrapper_type: str | None = None
    has_shape_spec: bool = False
    has_model_settings: bool = False
    dtype: torch.dtype | None = None


def extract_state_dict(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Extract a bare model state dict from a Lightning checkpoint payload."""
    if not isinstance(checkpoint, dict):
        return checkpoint
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(key.startswith("model.") for key in state_dict):
        return state_dict
    logger.debug("Stripping 'model.' prefix from state dict keys")
    return {
        key.replace("model.", "", 1) if key.startswith("model.") else key: value
        for key, value in state_dict.items()
    }


def extract_model_settings(checkpoint: dict[str, Any]) -> ModelComponentSettings:
    """Reconstruct model settings from checkpoint metadata."""
    if "dlkit_metadata" not in checkpoint:
        raise WorkflowError("Checkpoint missing 'dlkit_metadata'.")
    if "model_settings" not in checkpoint["dlkit_metadata"]:
        raise WorkflowError("Checkpoint 'dlkit_metadata' missing 'model_settings'.")

    try:
        from dlkit.runtime.adapters.lightning.checkpoint_dto import normalize_checkpoint_metadata

        metadata = normalize_checkpoint_metadata(checkpoint["dlkit_metadata"])
        settings_data = metadata["model_settings"]
        if "resolved_init_kwargs" in settings_data:
            reconstructed = {
                "name": settings_data.get("name") or "",
                "module_path": settings_data.get("module_path") or "dlkit.domain.nn",
                **(settings_data.get("resolved_init_kwargs") or {}),
            }
            return ModelComponentSettings.model_validate(reconstructed)
        return ModelComponentSettings.model_validate(settings_data)
    except WorkflowError:
        raise
    except Exception as exc:
        raise WorkflowError(f"Failed to deserialize model settings: {exc}") from exc


def detect_checkpoint_dtype(state_dict: dict[str, Any]) -> torch.dtype:
    """Infer a checkpoint dtype from the first floating tensor parameter."""
    for value in state_dict.values():
        if isinstance(value, torch.Tensor) and value.dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            logger.debug("Detected checkpoint dtype: {}", value.dtype)
            return value.dtype
    logger.warning("Could not detect dtype from checkpoint, defaulting to float32")
    return torch.float32


def load_checkpoint(checkpoint_path: Path | str) -> dict[str, Any]:
    """Load a checkpoint file from disk."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        logger.debug("Loading checkpoint from {}", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise WorkflowError(f"Invalid checkpoint format: expected dict, got {type(checkpoint)}")
        return checkpoint
    except Exception as exc:
        raise WorkflowError(f"Failed to load checkpoint from {checkpoint_path}: {exc}") from exc


def validate_checkpoint(checkpoint_path: Path | str) -> CheckpointValidationResult:
    """Validate checkpoint structure and return lightweight metadata."""
    checkpoint_path = Path(checkpoint_path)
    exists = checkpoint_path.exists()
    if not exists:
        return CheckpointValidationResult(
            checkpoint_path=checkpoint_path,
            exists=False,
            valid_format=False,
            has_state_dict=False,
            has_model_settings=False,
            has_shape_metadata=False,
            dtype=None,
        )

    valid_format = False
    has_state_dict = False
    has_model_settings = False
    has_shape_metadata = False
    dtype = None
    try:
        checkpoint = load_checkpoint(checkpoint_path)
        valid_format = True
        state_dict = extract_state_dict(checkpoint)
        if state_dict:
            has_state_dict = True
            dtype = detect_checkpoint_dtype(state_dict)
        try:
            extract_model_settings(checkpoint)
            has_model_settings = True
        except WorkflowError:
            pass
        if "dlkit_metadata" in checkpoint and "shape_summary" in checkpoint["dlkit_metadata"]:
            has_shape_metadata = True
    except Exception as exc:
        logger.error("Checkpoint validation failed: {}", exc)

    return CheckpointValidationResult(
        checkpoint_path=checkpoint_path,
        exists=exists,
        valid_format=valid_format,
        has_state_dict=has_state_dict,
        has_model_settings=has_model_settings,
        has_shape_metadata=has_shape_metadata,
        dtype=dtype,
    )


def get_checkpoint_info(checkpoint_path: Path | str) -> CheckpointInfo:
    """Extract checkpoint metadata without constructing the model."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)
    has_dlkit_metadata = "dlkit_metadata" in checkpoint
    has_hyper_parameters = "hyper_parameters" in checkpoint
    model_family = None
    wrapper_type = None
    has_shape_spec = False
    has_model_settings = False
    dtype = None

    if has_dlkit_metadata:
        metadata = checkpoint["dlkit_metadata"]
        model_family = metadata.get("model_family")
        wrapper_type = metadata.get("wrapper_type")
        has_shape_spec = "shape_summary" in metadata
        has_model_settings = "model_settings" in metadata

    state_dict = extract_state_dict(checkpoint)
    if state_dict:
        dtype = detect_checkpoint_dtype(state_dict)

    return CheckpointInfo(
        checkpoint_path=checkpoint_path,
        has_dlkit_metadata=has_dlkit_metadata,
        has_hyper_parameters=has_hyper_parameters,
        model_family=model_family,
        wrapper_type=wrapper_type,
        has_shape_spec=has_shape_spec,
        has_model_settings=has_model_settings,
        dtype=dtype,
    )
