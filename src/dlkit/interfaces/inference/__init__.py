"""Inference system for DLKit.

This module provides a complete inference solution that operates
independently from training configurations, using only model checkpoints
and arbitrary inputs.

Key Components:
- InferenceConfig: Pure inference configuration
- InferenceInput: Flexible input handling for tensors, dicts, arrays, files
- TransformChainExecutor: Standalone transform application
- InferenceStrategy: Direct model.forward() execution
- InferenceService: High-level inference orchestration

Example:
    >>> from dlkit.interfaces.inference import infer
    >>> result = infer(
    ...     checkpoint_path="model.ckpt",
    ...     inputs={"x": torch.randn(32, 10)},
    ...     batch_size=16
    ... )
    >>> predictions = result.predictions
"""

from .config.inference_config import InferenceConfig
from .inputs.inference_input import InferenceInput
from .service import InferenceService
from .api import infer, predict

__all__ = [
    "InferenceConfig",
    "InferenceInput",
    "InferenceService",
    "infer",
    "predict",
]