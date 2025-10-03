"""CLI adapters for configuration and result presentation."""

from .config_adapter import load_config, validate_config_path
from .result_presenter import (
    present_inference_result,
    present_optimization_result,
    present_training_result,
)

__all__ = [
    # Config loading
    "load_config",
    "validate_config_path",
    # Result presentation
    "present_training_result",
    "present_inference_result",
    "present_optimization_result",
]
