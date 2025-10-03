"""Flexible input handling for inference.

This module provides a unified input system that can handle various input
formats (tensors, dicts, arrays, files) and convert them to the format
expected by DLKit models.
"""

from .inference_input import InferenceInput
from .adapters import InputAdapter, TensorInputAdapter, DictInputAdapter, ArrayInputAdapter, FileInputAdapter

__all__ = [
    "InferenceInput",
    "InputAdapter",
    "TensorInputAdapter",
    "DictInputAdapter",
    "ArrayInputAdapter",
    "FileInputAdapter",
]