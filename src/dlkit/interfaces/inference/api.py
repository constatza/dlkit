"""Inference API functions.

This module provides the new API functions for inference
that replace the existing inference system with breaking changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .service import InferenceService
from .inputs.inference_input import InferenceInput
from .config.inference_config import InferenceConfig
from dlkit.tools.config import GeneralSettings
from dlkit.interfaces.api.domain.models import InferenceResult


# Global inference service instance
_inference_service = InferenceService()


def infer(
    checkpoint_path: Path | str,
    inputs: InferenceInput | dict[str, Any] | Any,
    device: str = "auto",
    batch_size: int = 32,
    apply_transforms: bool = True
) -> InferenceResult:
    """Execute shape-free inference from checkpoint only.

    This is the primary KISS API for inference that eliminates manual shape
    parameters. It automatically reconstructs models from enhanced checkpoint
    metadata without requiring training configuration files, datasets, or
    manual shape specifications.

    Model shapes are automatically inferred from checkpoint metadata saved
    during training. Batch dimensions are handled automatically.

    Args:
        checkpoint_path: Path to trained model checkpoint
        inputs: Input data (InferenceInput or raw data for auto-conversion)
        device: Device specification ("auto", "cpu", "cuda", "mps")
        batch_size: Batch size for processing
        apply_transforms: Whether to apply fitted transforms

    Returns:
        InferenceResult: Inference execution result

    Raises:
        WorkflowError: On inference execution failure

    Example:
        >>> import torch
        >>> from dlkit.interfaces.inference import infer
        >>>
        >>> # That's it! No shapes, no config files needed
        >>> result = infer("model.ckpt", {"x": torch.randn(32, 10)})
        >>> predictions = result.predictions
        >>>
        >>> # Works with single tensors too
        >>> result = infer("model.ckpt", torch.randn(32, 10))
        >>>
        >>> # Or from files
        >>> result = infer("model.ckpt", "data.npy")

    Note:
        This function automatically infers model shapes from checkpoint metadata.
        For checkpoints saved with DLKit v2.0+, shape inference is fully automatic.
        Legacy checkpoints may require fallback strategies or manual conversion.
    """
    # Use the new architecture via dependency injection container
    from .container import get_inference_orchestrator

    orchestrator = get_inference_orchestrator()
    return orchestrator.infer_from_checkpoint(
        checkpoint_path=checkpoint_path,
        inputs=inputs,
        device=device,
        batch_size=batch_size,
        apply_transforms=apply_transforms
    )


def infer_with_config(
    config: InferenceConfig,
    inputs: InferenceInput | dict[str, Any] | Any,
    batch_size: int | None = None
) -> InferenceResult:
    """Execute inference with pre-built configuration.

    Args:
        config: Inference configuration
        inputs: Input data (InferenceInput or raw data for auto-conversion)
        batch_size: Batch size for processing (None = use config default)

    Returns:
        InferenceResult: Inference execution result

    Example:
        >>> from dlkit.interfaces.inference.config import build_inference_config_from_checkpoint
        >>> config = build_inference_config_from_checkpoint("model.ckpt", device="cuda")
        >>> result = infer_with_config(config, {"x": torch.randn(32, 10)})
    """
    # Auto-convert inputs to InferenceInput if needed
    if not isinstance(inputs, InferenceInput):
        inputs = InferenceInput(inputs)

    return _inference_service.infer_with_config(
        config=config,
        inputs=inputs,
        batch_size=batch_size
    )


def predict(
    training_settings: GeneralSettings,
    checkpoint_path: Path | str,
    **overrides: Any
) -> InferenceResult:
    """Execute simple prediction using Lightning framework.

    This method uses the traditional Lightning-based inference approach
    for validation and testing scenarios where training configuration
    and datasets are available.

    Args:
        training_settings: Training workflow settings (BREAKING: now required)
        checkpoint_path: Path to model checkpoint
        **overrides: Additional parameter overrides

    Returns:
        InferenceResult: Inference execution result

    Raises:
        WorkflowError: On inference execution failure

    Example:
        >>> from dlkit.tools.config import load_training_settings
        >>> settings = load_training_settings("config.toml")
        >>> result = predict(settings, "model.ckpt", batch_size=64)

    Note:
        This replaces the old infer() function which previously accepted
        InferenceWorkflowSettings. Now only TrainingWorkflowSettings
        are supported for prediction mode.
    """
    return _inference_service.predict(
        training_settings=training_settings,
        checkpoint_path=checkpoint_path,
        **overrides
    )


def validate_checkpoint(checkpoint_path: Path | str) -> dict[str, str]:
    """Validate checkpoint compatibility for inference.

    Args:
        checkpoint_path: Path to model checkpoint

    Returns:
        Dictionary of validation errors (empty if valid)

    Example:
        >>> errors = validate_checkpoint("model.ckpt")
        >>> if not errors:
        ...     print("Checkpoint is compatible with inference")
        >>> else:
        ...     print(f"Validation errors: {errors}")
    """
    return _inference_service.validate_checkpoint(checkpoint_path)


def get_checkpoint_info(checkpoint_path: Path | str) -> dict[str, Any]:
    """Get information about a checkpoint file.

    Args:
        checkpoint_path: Path to model checkpoint

    Returns:
        Dictionary with checkpoint information

    Example:
        >>> info = get_checkpoint_info("model.ckpt")
        >>> print(f"Features: {info['inference_metadata']['feature_names']}")
        >>> print(f"Targets: {info['inference_metadata']['target_names']}")
        >>> print(f"Has transforms: {info['has_transforms']}")
    """
    return _inference_service.get_checkpoint_info(checkpoint_path)


# Convenience functions for common use cases

def infer_from_tensors(
    checkpoint_path: Path | str,
    tensors: dict[str, Any],
    device: str = "auto",
    batch_size: int = 32
) -> InferenceResult:
    """Convenience function for tensor-based inference.

    Args:
        checkpoint_path: Path to model checkpoint
        tensors: Dictionary of input tensors
        device: Device specification
        batch_size: Batch size for processing

    Returns:
        InferenceResult: Inference execution result
    """
    return infer(checkpoint_path, tensors, device, batch_size)


def infer_from_arrays(
    checkpoint_path: Path | str,
    arrays: dict[str, Any],
    device: str = "auto",
    batch_size: int = 32
) -> InferenceResult:
    """Convenience function for NumPy array-based inference.

    Args:
        checkpoint_path: Path to model checkpoint
        arrays: Dictionary of input NumPy arrays
        device: Device specification
        batch_size: Batch size for processing

    Returns:
        InferenceResult: Inference execution result
    """
    return infer(checkpoint_path, arrays, device, batch_size)


def infer_from_file(
    checkpoint_path: Path | str,
    file_path: Path | str,
    device: str = "auto",
    batch_size: int = 32
) -> InferenceResult:
    """Convenience function for file-based inference.

    Args:
        checkpoint_path: Path to model checkpoint
        file_path: Path to input data file
        device: Device specification
        batch_size: Batch size for processing

    Returns:
        InferenceResult: Inference execution result
    """
    return infer(checkpoint_path, file_path, device, batch_size)