"""Checkpoint and model loading utilities.

Consolidated loading logic without hexagonal architecture overhead.
Direct functions replacing use cases, adapters, and builders.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dlkit.core.shape_specs.simple_inference import ShapeSummary

import torch
from loguru import logger

from dlkit.interfaces.api.domain.errors import WorkflowError
from dlkit.tools.config.components.model_components import ModelComponentSettings

from .shapes import infer_shape_specification


@dataclass
class CheckpointValidationResult:
    """Result of checkpoint validation with type-safe fields.

    Attributes:
        checkpoint_path: Path to the checkpoint file.
        exists: Whether the checkpoint file exists.
        valid_format: Whether the checkpoint has valid format.
        has_state_dict: Whether state dict is present.
        has_model_settings: Whether model settings are present.
        has_shape_metadata: Whether shape metadata is present.
        dtype: Detected dtype from state dict (None if unknown).
    """

    checkpoint_path: Path
    exists: bool
    valid_format: bool
    has_state_dict: bool
    has_model_settings: bool
    has_shape_metadata: bool
    dtype: torch.dtype | None


@dataclass
class CheckpointInfo:
    """Checkpoint metadata information with type-safe fields.

    Attributes:
        checkpoint_path: Path to the checkpoint file.
        has_dlkit_metadata: Whether dlkit_metadata is present.
        has_hyper_parameters: Whether hyper_parameters is present.
        version: Version string from metadata (None if not present).
        model_family: Model family from metadata (None if not present).
        wrapper_type: Wrapper type from metadata (None if not present).
        has_shape_spec: Whether shape_spec is present.
        has_model_settings: Whether model_settings are present.
        dtype: Detected dtype from state dict (None if unknown).
    """

    checkpoint_path: Path
    has_dlkit_metadata: bool
    has_hyper_parameters: bool
    version: str | None = None
    model_family: str | None = None
    wrapper_type: str | None = None
    has_shape_spec: bool = False
    has_model_settings: bool = False
    dtype: torch.dtype | None = None


def extract_state_dict(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Extract state dict from checkpoint with automatic prefix stripping.

    Args:
        checkpoint: Loaded checkpoint dictionary

    Returns:
        State dict with 'model.' prefix stripped if present
    """
    # Extract raw state dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        return state_dict

    # Check if 'model.' prefix exists
    has_prefix = any(k.startswith("model.") for k in state_dict.keys())

    if has_prefix:
        logger.info("Stripping 'model.' prefix from state dict keys")
        return {
            k.replace("model.", "", 1) if k.startswith("model.") else k: v
            for k, v in state_dict.items()
        }

    return state_dict


def extract_model_settings(checkpoint: dict[str, Any]) -> ModelComponentSettings:
    """Extract model settings from checkpoint metadata.

    Args:
        checkpoint: Loaded checkpoint dictionary

    Returns:
        ModelComponentSettings reconstructed from checkpoint

    Raises:
        WorkflowError: If model settings cannot be extracted
    """
    # Try enhanced metadata first
    if "dlkit_metadata" in checkpoint and "model_settings" in checkpoint["dlkit_metadata"]:
        try:
            settings_data = checkpoint["dlkit_metadata"]["model_settings"]
            return ModelComponentSettings.model_validate(settings_data)
        except Exception as e:
            raise WorkflowError(
                f"Failed to deserialize model settings from enhanced metadata: {e}",
                {"has_dlkit_metadata": "true"},
            ) from e

    # Try legacy hyper_parameters
    if "hyper_parameters" in checkpoint and "model_settings" in checkpoint["hyper_parameters"]:
        try:
            settings_data = checkpoint["hyper_parameters"]["model_settings"]
            return ModelComponentSettings.model_validate(settings_data)
        except Exception as e:
            raise WorkflowError(
                f"Failed to deserialize model settings from legacy checkpoint: {e}",
                {"has_hyper_parameters": "true"},
            ) from e

    raise WorkflowError(
        "Cannot extract model settings: Checkpoint missing both enhanced metadata and hyper_parameters",
        {
            "has_dlkit_metadata": str("dlkit_metadata" in checkpoint),
            "has_hyper_parameters": str("hyper_parameters" in checkpoint),
        },
    )


def detect_checkpoint_dtype(state_dict: dict[str, Any]) -> torch.dtype:
    """Detect dtype from checkpoint state dict.

    Args:
        state_dict: Model state dictionary

    Returns:
        torch.dtype detected from parameters (defaults to float32)
    """
    # Find first tensor parameter to detect dtype
    for value in state_dict.values():
        if isinstance(value, torch.Tensor) and value.dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            logger.info(f"Detected checkpoint dtype: {value.dtype}")
            return value.dtype

    # Default to float32
    logger.warning("Could not detect dtype from checkpoint, defaulting to float32")
    return torch.float32


def build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    shape_spec: "ShapeSummary | None" = None,
) -> torch.nn.Module:
    """Build and load model from checkpoint.

    This function consolidates:
    - Model settings extraction
    - Model instantiation via build_model() pure factory
    - Dtype detection and conversion
    - State dict loading

    Args:
        checkpoint: Loaded checkpoint dictionary
        shape_spec: Shape specification/summary for model construction (optional)

    Returns:
        Loaded PyTorch model in eval mode

    Raises:
        WorkflowError: If model building fails
    """
    import importlib
    from dlkit.core.models.factory import build_model as _build_model

    # Extract model settings
    model_settings = extract_model_settings(checkpoint)

    # Extract and prepare state dict
    state_dict = extract_state_dict(checkpoint)
    checkpoint_dtype = detect_checkpoint_dtype(state_dict)

    # shape_spec may be a ShapeSummary (from new format) or None
    shape_summary = shape_spec if shape_spec is not None else None
    logger.info(f"Building model with shape summary: {shape_summary}")

    # Resolve model class — use directly if already a type, else import
    if isinstance(model_settings.name, type):
        model_cls = model_settings.name
    else:
        try:
            module = importlib.import_module(model_settings.module_path)
            model_cls = getattr(module, model_settings.name)
        except (ImportError, AttributeError) as e:
            raise WorkflowError(
                f"Failed to import model class {model_settings.module_path}.{model_settings.name}: {e}"
            ) from e

    # Build model using pure factory (tries in_features, in_channels, then kwargs-only)
    # _serialize_model_settings nests hyperparams under 'params'; unpack them here.
    if hasattr(model_settings, "model_dump"):
        all_fields = model_settings.model_dump()
        excluded = {"name", "module_path", "checkpoint", "class_name"}
        params_nested = all_fields.pop("params", None) or {}
        hyperparams = {k: v for k, v in all_fields.items() if k not in excluded and v is not None}
        hyperparams.update(params_nested)
    else:
        hyperparams = {}
    model = _build_model(model_cls, shape_summary, hyperparams)

    # Convert model to checkpoint dtype BEFORE loading weights
    # This prevents precision loss during state dict loading
    logger.info(f"Converting model to checkpoint dtype: {checkpoint_dtype}")
    model = model.to(dtype=checkpoint_dtype)

    # Load state dict
    try:
        model.load_state_dict(state_dict, strict=False)
        logger.info("Successfully loaded model weights from checkpoint")
    except Exception as e:
        raise WorkflowError(
            f"Failed to load state dict into model: {e}", {"model_type": type(model).__name__}
        ) from e

    # Set to eval mode
    model.eval()

    return model


def load_checkpoint(checkpoint_path: Path | str) -> dict[str, Any]:
    """Load checkpoint from disk.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Loaded checkpoint dictionary

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        WorkflowError: If loading fails
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if not isinstance(checkpoint, dict):
            raise WorkflowError(f"Invalid checkpoint format: expected dict, got {type(checkpoint)}")

        return checkpoint

    except Exception as e:
        raise WorkflowError(f"Failed to load checkpoint from {checkpoint_path}: {e}") from e


def validate_checkpoint(checkpoint_path: Path | str) -> CheckpointValidationResult:
    """Validate checkpoint and return metadata.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Type-safe validation result with bool fields and proper dtype

    Raises:
        WorkflowError: If validation fails critically
    """
    checkpoint_path = Path(checkpoint_path)

    # Initialize with default values
    result = CheckpointValidationResult(
        checkpoint_path=checkpoint_path,
        exists=checkpoint_path.exists(),
        valid_format=False,
        has_state_dict=False,
        has_model_settings=False,
        has_shape_metadata=False,
        dtype=None,
    )

    # Guard: Early return if file doesn't exist
    if not result.exists:
        return result

    try:
        checkpoint = load_checkpoint(checkpoint_path)
        result.valid_format = True

        # Check for state dict
        state_dict = extract_state_dict(checkpoint)
        if state_dict:
            result.has_state_dict = True
            result.dtype = detect_checkpoint_dtype(state_dict)

        # Check for model settings
        try:
            extract_model_settings(checkpoint)
            result.has_model_settings = True
        except WorkflowError:
            pass

        # Check for shape metadata
        if "dlkit_metadata" in checkpoint and "shape_summary" in checkpoint["dlkit_metadata"]:
            result.has_shape_metadata = True

    except Exception as e:
        logger.error(f"Checkpoint validation failed: {e}")

    return result


def get_checkpoint_info(checkpoint_path: Path | str) -> CheckpointInfo:
    """Extract metadata from checkpoint without loading model.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Type-safe checkpoint information with proper types
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)

    # Initialize with required fields
    info = CheckpointInfo(
        checkpoint_path=checkpoint_path,
        has_dlkit_metadata="dlkit_metadata" in checkpoint,
        has_hyper_parameters="hyper_parameters" in checkpoint,
    )

    # Extract dlkit metadata if present
    if "dlkit_metadata" in checkpoint:
        metadata = checkpoint["dlkit_metadata"]
        info.version = metadata.get("version")
        info.model_family = metadata.get("model_family")
        info.wrapper_type = metadata.get("wrapper_type")
        info.has_shape_spec = "shape_summary" in metadata
        info.has_model_settings = "model_settings" in metadata

    # Extract dtype info
    state_dict = extract_state_dict(checkpoint)
    if state_dict:
        info.dtype = detect_checkpoint_dtype(state_dict)

    return info
