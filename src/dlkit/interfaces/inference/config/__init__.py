"""Inference configuration system.

This module provides configuration classes for inference that are
completely independent from training configurations.
"""

from .inference_config import InferenceConfig
from .config_builder import InferenceConfigBuilder, build_inference_config_from_checkpoint

__all__ = [
    "InferenceConfig",
    "InferenceConfigBuilder",
    "build_inference_config_from_checkpoint",
]