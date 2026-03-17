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

from dlkit.interfaces.api.domain.errors import WorkflowError
from dlkit.tools.config.components.model_components import ModelComponentSettings
from dlkit.tools.utils.logging_config import get_logger

from .shapes import infer_shape_specification

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
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


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointInfo:
    """Checkpoint metadata information with type-safe fields.

    Attributes:
        checkpoint_path: Path to the checkpoint file.
        has_dlkit_metadata: Whether dlkit_metadata is present.
        has_hyper_parameters: Whether hyper_parameters is present.
        model_family: Model family from metadata (None if not present).
        wrapper_type: Wrapper type from metadata (None if not present).
        has_shape_spec: Whether shape_spec is present.
        has_model_settings: Whether model_settings are present.
        dtype: Detected dtype from state dict (None if unknown).
    """

    checkpoint_path: Path
    has_dlkit_metadata: bool
    has_hyper_parameters: bool
    model_family: str | None = None
    wrapper_type: str | None = None
    has_shape_spec: bool = False
    has_model_settings: bool = False
    dtype: torch.dtype | None = None


def extract_state_dict(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Extract state dict from checkpoint with automatic prefix stripping.

    Reads ``checkpoint["state_dict"]`` and strips the ``"model."`` prefix
    that Lightning adds when saving ``LightningModule`` state.

    Args:
        checkpoint: Loaded checkpoint dictionary.

    Returns:
        State dict with ``"model."`` prefix stripped from all keys that carry it.
    """
    if not isinstance(checkpoint, dict):
        return checkpoint

    state_dict = checkpoint.get("state_dict", checkpoint)

    if not isinstance(state_dict, dict):
        return state_dict

    if not any(k.startswith("model.") for k in state_dict):
        return state_dict

    logger.debug("Stripping 'model.' prefix from state dict keys")
    return {
        k.replace("model.", "", 1) if k.startswith("model.") else k: v
        for k, v in state_dict.items()
    }


def extract_model_settings(checkpoint: dict[str, Any]) -> ModelComponentSettings:
    """Extract model settings from checkpoint metadata.

    Args:
        checkpoint: Loaded checkpoint dictionary.

    Returns:
        ModelComponentSettings reconstructed from checkpoint.

    Raises:
        WorkflowError: If model settings cannot be extracted.
    """
    if "dlkit_metadata" not in checkpoint:
        raise WorkflowError("Checkpoint missing 'dlkit_metadata'.")
    if "model_settings" not in checkpoint["dlkit_metadata"]:
        raise WorkflowError("Checkpoint 'dlkit_metadata' missing 'model_settings'.")

    try:
        from dlkit.core.models.wrappers.checkpoint_dto import normalize_checkpoint_metadata

        metadata = normalize_checkpoint_metadata(checkpoint["dlkit_metadata"])
        settings_data = metadata["model_settings"]

        # Flat DTO format: reconstruct from resolved_init_kwargs to avoid DTO-specific
        # fields (resolved_init_kwargs, all_hyperparams) leaking into the model constructor.
        if "resolved_init_kwargs" in settings_data:
            name = settings_data.get("name") or ""
            module_path = settings_data.get("module_path") or "dlkit.core.models.nn"
            init_kwargs = settings_data.get("resolved_init_kwargs") or {}
            reconstructed = {"name": name, "module_path": module_path, **init_kwargs}
            return ModelComponentSettings.model_validate(reconstructed)

        return ModelComponentSettings.model_validate(settings_data)
    except WorkflowError:
        raise
    except Exception as e:
        raise WorkflowError(f"Failed to deserialize model settings: {e}") from e


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
            logger.debug("Detected checkpoint dtype: {}", value.dtype)
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

    # Extract and prepare state dict.
    # Extract only model-prefixed keys for strict loading against the bare model class.
    # Wrapper-level keys (_batch_transformer, etc.) are discarded.
    raw_sd = checkpoint.get("state_dict", checkpoint)
    if isinstance(raw_sd, dict) and any(k.startswith("model.") for k in raw_sd):
        state_dict = {k[len("model."):]: v for k, v in raw_sd.items() if k.startswith("model.")}
    else:
        state_dict = raw_sd if isinstance(raw_sd, dict) else {}
    checkpoint_dtype = detect_checkpoint_dtype(state_dict)

    logger.debug("Building model with shape summary: {}", shape_spec)

    # Resolve model class — use directly if already a type, else import
    if isinstance(model_settings.name, type):
        model_cls = model_settings.name
    else:
        class_name = model_settings.name
        module_path = model_settings.module_path
        if not isinstance(class_name, str) or not module_path:
            raise WorkflowError(
                f"Cannot resolve model class: name={class_name!r}, module_path={module_path!r}"
            )
        try:
            module = importlib.import_module(module_path)
            model_cls = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise WorkflowError(
                f"Failed to import model class {module_path}.{class_name}: {e}"
            ) from e

    # Extract init kwargs using whitelist tag (no exclusion list needed)
    from dlkit.tools.config.components.model_components import extract_init_kwargs
    hyperparams = extract_init_kwargs(model_settings)
    model = _build_model(model_cls, shape_spec, hyperparams)

    # Convert model to checkpoint dtype BEFORE loading weights
    # This prevents precision loss during state dict loading
    logger.debug("Converting model to checkpoint dtype: {}", checkpoint_dtype)
    model = model.to(dtype=checkpoint_dtype)

    # Load state dict (strict=True to catch weight drift between save and load)
    try:
        model.load_state_dict(state_dict, strict=True)
        logger.debug("Successfully loaded model weights from checkpoint")
    except RuntimeError as e:
        # Re-raise with clear key listing for easier debugging
        raise WorkflowError(
            f"State dict mismatch loading {type(model).__name__}: {e}. "
            "If intentional partial loading is required, call load_state_dict(strict=False) directly.",
            {"model_type": type(model).__name__},
        ) from e
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
        logger.debug("Loading checkpoint from {}", checkpoint_path)
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

    exists = checkpoint_path.exists()

    # Guard: Early return if file doesn't exist
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

        # Check for state dict
        state_dict = extract_state_dict(checkpoint)
        if state_dict:
            has_state_dict = True
            dtype = detect_checkpoint_dtype(state_dict)

        # Check for model settings
        try:
            extract_model_settings(checkpoint)
            has_model_settings = True
        except WorkflowError:
            pass

        # Check for shape metadata
        if "dlkit_metadata" in checkpoint and "shape_summary" in checkpoint["dlkit_metadata"]:
            has_shape_metadata = True

    except Exception as e:
        logger.error(f"Checkpoint validation failed: {e}")

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
    """Extract metadata from checkpoint without loading model.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Type-safe checkpoint information with proper types
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)

    has_dlkit_metadata = "dlkit_metadata" in checkpoint
    has_hyper_parameters = "hyper_parameters" in checkpoint
    model_family = None
    wrapper_type = None
    has_shape_spec = False
    has_model_settings = False
    dtype = None

    # Extract dlkit metadata if present
    if has_dlkit_metadata:
        metadata = checkpoint["dlkit_metadata"]
        model_family = metadata.get("model_family")
        wrapper_type = metadata.get("wrapper_type")
        has_shape_spec = "shape_summary" in metadata
        has_model_settings = "model_settings" in metadata

    # Extract dtype info
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
