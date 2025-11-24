"""Inference API - New Stateful Predictor Architecture.

This module provides the new industry-standard inference API with stateful
predictors that can be loaded once and reused for multiple predictions.

Breaking Changes from Previous API:
- Removed: infer() function (was loading model on every call)
- Removed: infer_with_config() function
- Removed: infer_from_tensors(), infer_from_arrays(), infer_from_file()
- Removed: predict() Lightning-based function
- Added: load_predictor() - new primary API for efficient inference

Migration Guide:
    OLD (removed):
        from dlkit import infer
        result = infer("model.ckpt", inputs)

    NEW:
        from dlkit import load_predictor

        # One-shot inference:
        with load_predictor("model.ckpt") as predictor:
            result = predictor.predict(inputs)

        # Multiple inferences (efficient):
        predictor = load_predictor("model.ckpt")
        result1 = predictor.predict(input1)
        result2 = predictor.predict(input2)
        predictor.unload()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dlkit.tools.config.precision.strategy import PrecisionStrategy
from .predictor import CheckpointPredictor
from .factory import PredictorFactory


def load_predictor(
    checkpoint_path: Path | str,
    device: str = "auto",
    batch_size: int = 32,
    apply_transforms: bool = True,
    auto_load: bool = True,
    precision: PrecisionStrategy | None = None
) -> CheckpointPredictor:
    """Load a stateful predictor from checkpoint (NEW PRIMARY API).

    This is the recommended way to perform inference. The predictor loads
    the model ONCE and can be reused for multiple predictions without
    reloading the checkpoint.

    For iterative workflows (100+ predictions), this provides 10-100x
    performance improvement compared to the old infer() function.

    Args:
        checkpoint_path: Path to trained model checkpoint
        device: Device specification:
            - "auto": Automatically select (CUDA > MPS > CPU)
            - "cpu": Force CPU
            - "cuda" or "cuda:0": Use specific CUDA device
            - "mps": Use Apple Silicon GPU
        batch_size: Default batch size for inference
        apply_transforms: Whether to apply fitted transforms from checkpoint
        auto_load: If True, load model immediately (recommended)
        precision: Optional precision override for data loading.
                  If None, infers precision from model checkpoint dtype.
                  This ensures input data dtype matches model parameter dtype.

    Returns:
        CheckpointPredictor: Reusable predictor object

    Raises:
        WorkflowError: If checkpoint loading fails

    Examples:
        >>> # Basic usage (one-shot inference)
        >>> from dlkit import load_predictor
        >>> with load_predictor("model.ckpt") as predictor:
        ...     result = predictor.predict({"x": torch.randn(32, 10)})

        >>> # Efficient multi-inference
        >>> predictor = load_predictor("model.ckpt", device="cuda")
        >>> for data in dataset:  # No reloading!
        ...     result = predictor.predict(data)
        ...     process(result)
        >>> predictor.unload()

        >>> # Config-based batch inference
        >>> predictor = load_predictor("model.ckpt")
        >>> for batch_result in predictor.predict_from_config("config.toml"):
        ...     process(batch_result)

        >>> # Lazy loading
        >>> predictor = load_predictor("model.ckpt", auto_load=False)
        >>> # ... do other setup ...
        >>> predictor.load()  # Explicit load
        >>> result = predictor.predict(inputs)

        >>> # Explicit precision override
        >>> from dlkit.tools.config.precision.strategy import PrecisionStrategy
        >>> predictor = load_predictor("model.ckpt", precision=PrecisionStrategy.FULL_64)

    Note:
        The predictor encapsulates:
        - Loaded PyTorch model (in eval mode)
        - Transform executor (if applicable)
        - Device placement
        - Inference configuration
        - Precision context (inferred from model or explicitly set)

        All expensive operations (checkpoint loading, weight loading, device
        placement) happen ONCE during load(), not on every predict() call.

        Precision handling: By default, the predictor infers precision from
        the loaded model's parameter dtype. This ensures data is loaded with
        matching precision to avoid dtype mismatch errors. You can override
        this with the precision parameter if needed.
    """
    from .container import get_predictor_factory

    factory = get_predictor_factory()
    return factory.create_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        batch_size=batch_size,
        apply_transforms=apply_transforms,
        auto_load=auto_load,
        precision=precision
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
    import torch
    from pathlib import Path

    errors = {}
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        errors["file"] = "Checkpoint file not found"
        return errors

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if not isinstance(checkpoint, dict):
            errors["format"] = "Invalid checkpoint format (expected dict)"
            return errors

        if 'dlkit_metadata' not in checkpoint:
            errors["metadata"] = "Checkpoint missing 'dlkit_metadata' (legacy format not supported)"
            return errors

        metadata = checkpoint['dlkit_metadata']

        if metadata.get('version') != '2.0':
            errors["version"] = f"Unsupported version '{metadata.get('version')}' (only '2.0' supported)"

        if 'model_settings' not in metadata:
            errors["model_settings"] = "Checkpoint metadata missing 'model_settings'"

    except Exception as e:
        errors["load"] = f"Failed to load checkpoint: {str(e)}"

    return errors


def get_checkpoint_info(checkpoint_path: Path | str) -> dict[str, Any]:
    """Get information about a checkpoint file.

    Args:
        checkpoint_path: Path to model checkpoint

    Returns:
        Dictionary with checkpoint information

    Raises:
        WorkflowError: If checkpoint cannot be read

    Example:
        >>> info = get_checkpoint_info("model.ckpt")
        >>> print(f"Model: {info['model_name']}")
        >>> print(f"Has transforms: {info['has_transforms']}")
        >>> print(f"Shape: {info['shape_info']}")
    """
    import torch
    from pathlib import Path
    from dlkit.interfaces.api.domain.errors import WorkflowError

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise WorkflowError(
            f"Checkpoint not found: {checkpoint_path}",
            {"function": "get_checkpoint_info"}
        )

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        info = {
            "checkpoint_path": str(checkpoint_path),
            "file_size_mb": checkpoint_path.stat().st_size / (1024 * 1024),
            "has_dlkit_metadata": 'dlkit_metadata' in checkpoint,
            "has_state_dict": 'state_dict' in checkpoint,
        }

        if 'dlkit_metadata' in checkpoint:
            metadata = checkpoint['dlkit_metadata']
            info["version"] = metadata.get('version')
            info["has_transforms"] = 'transforms' in checkpoint

            if 'model_settings' in metadata:
                model_settings = metadata['model_settings']
                info["model_name"] = model_settings.get('name')
                info["model_module"] = model_settings.get('module_path')

            if 'shape_spec' in metadata:
                info["shape_info"] = metadata['shape_spec']

        return info

    except Exception as e:
        raise WorkflowError(
            f"Failed to read checkpoint info: {str(e)}",
            {"function": "get_checkpoint_info", "checkpoint": str(checkpoint_path)}
        ) from e


# Export main functions
__all__ = [
    "load_predictor",
    "validate_checkpoint",
    "get_checkpoint_info",
]
