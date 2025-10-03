"""Inference strategies for different execution modes.

This module provides strategy implementations for executing inference
in different contexts - production (direct) and prediction (Lightning-based).
"""

from .inference_strategy import InferenceStrategy
from .prediction_strategy import SimplePredictionStrategy

__all__ = [
    "InferenceStrategy",
    "SimplePredictionStrategy",
]