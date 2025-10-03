"""Flexible input wrapper for inference.

This module provides a unified input interface that can handle various
input formats and convert them to the tensor format expected by DLKit models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import torch
import numpy as np
from torch import Tensor

from .adapters import InputAdapter, TensorInputAdapter, DictInputAdapter, ArrayInputAdapter, FileInputAdapter


# Type alias for supported input types
InputType = Union[
    Tensor,                           # Single tensor
    dict[str, Tensor],               # Dict of tensors
    dict[str, np.ndarray],           # Dict of arrays
    dict[str, Any],                  # Mixed dict
    np.ndarray,                      # Single array
    Path,                            # File path
    str,                             # File path as string
    list[dict[str, Any]],           # Batch of dicts
]


class InferenceInput:
    """Flexible input wrapper for inference.

    This class provides a unified interface for handling various input formats
    commonly used in inference scenarios. It automatically detects
    the input type and converts it to the standardized tensor dictionary format
    expected by DLKit models.

    Supported input types:
    - Single tensors: torch.Tensor
    - Dictionary of tensors: dict[str, torch.Tensor]
    - Dictionary of arrays: dict[str, np.ndarray]
    - NumPy arrays: np.ndarray
    - File paths: Path or str
    - Batch of dictionaries: list[dict[str, Any]]

    Example:
        >>> # Tensor input
        >>> input1 = InferenceInput(torch.randn(32, 10))
        >>> tensors1 = input1.to_tensor_dict(feature_names=["x"])

        >>> # Dict input
        >>> input2 = InferenceInput({"x": torch.randn(32, 10), "y": torch.randn(32, 5)})
        >>> tensors2 = input2.to_tensor_dict()

        >>> # Array input
        >>> input3 = InferenceInput({"features": np.random.randn(32, 10)})
        >>> tensors3 = input3.to_tensor_dict()
    """

    def __init__(self, data: InputType) -> None:
        """Initialize inference input wrapper.

        Args:
            data: Input data in any supported format

        Raises:
            ValueError: If input type is not supported
        """
        self._raw_data = data
        self._adapter: InputAdapter = self._select_adapter(data)

    def _select_adapter(self, data: InputType) -> InputAdapter:
        """Select appropriate adapter for the input data type.

        Args:
            data: Input data

        Returns:
            InputAdapter: Appropriate adapter for the data type

        Raises:
            ValueError: If no suitable adapter is found
        """
        # Tensor input
        if isinstance(data, Tensor):
            return TensorInputAdapter()

        # Dictionary input (tensors, arrays, or mixed)
        elif isinstance(data, dict):
            return DictInputAdapter()

        # NumPy array input
        elif isinstance(data, np.ndarray):
            return ArrayInputAdapter()

        # File path input
        elif isinstance(data, (Path, str)):
            return FileInputAdapter()

        # Batch input (list of dicts)
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            return DictInputAdapter()

        else:
            raise ValueError(
                f"Unsupported input type: {type(data)}. "
                f"Supported types: torch.Tensor, dict, np.ndarray, Path/str, list[dict]"
            )

    def to_tensor_dict(
        self,
        feature_names: list[str] | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32
    ) -> dict[str, Tensor]:
        """Convert input to standardized tensor dictionary format.

        Args:
            feature_names: Expected feature names for single tensor inputs
            device: Target device for tensors
            dtype: Target dtype for tensors

        Returns:
            Dictionary mapping feature names to tensors

        Raises:
            ValueError: If conversion fails or input is invalid
        """
        return self._adapter.convert_to_tensor_dict(
            self._raw_data,
            feature_names=feature_names,
            device=device,
            dtype=dtype
        )

    def get_batch_size(self) -> int | None:
        """Get the batch size of the input data.

        Returns:
            Batch size if determinable, None otherwise
        """
        try:
            tensor_dict = self.to_tensor_dict()
            if tensor_dict:
                first_tensor = next(iter(tensor_dict.values()))
                return first_tensor.shape[0]
        except Exception:
            pass
        return None

    def get_feature_names(self) -> list[str]:
        """Get the feature names from the input data.

        Returns:
            List of feature names (empty for single tensors)
        """
        if isinstance(self._raw_data, dict):
            return list(self._raw_data.keys())
        else:
            return []

    def validate_structure(self, expected_features: list[str] | None = None) -> dict[str, str]:
        """Validate input structure against expected features.

        Args:
            expected_features: List of expected feature names

        Returns:
            Dictionary of validation errors (empty if valid)
        """
        errors = {}

        try:
            # Convert to tensor dict to validate structure
            tensor_dict = self.to_tensor_dict()

            if expected_features:
                provided_features = set(tensor_dict.keys())
                expected_features_set = set(expected_features)

                missing_features = expected_features_set - provided_features
                if missing_features:
                    errors["missing_features"] = f"Missing features: {list(missing_features)}"

                extra_features = provided_features - expected_features_set
                if extra_features:
                    errors["extra_features"] = f"Unexpected features: {list(extra_features)}"

            # Validate tensor shapes are consistent
            batch_sizes = [tensor.shape[0] for tensor in tensor_dict.values() if tensor.ndim > 0]
            if batch_sizes and len(set(batch_sizes)) > 1:
                errors["inconsistent_batch_sizes"] = f"Inconsistent batch sizes: {batch_sizes}"

        except Exception as e:
            errors["conversion_failed"] = f"Failed to convert input: {e}"

        return errors

    def get_raw_data(self) -> InputType:
        """Get the original raw input data.

        Returns:
            Original input data in its native format
        """
        return self._raw_data

    def get_adapter_type(self) -> str:
        """Get the name of the adapter being used.

        Returns:
            Name of the adapter class
        """
        return self._adapter.__class__.__name__

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InferenceInput:
        """Create InferenceInput from dictionary data.

        Args:
            data: Dictionary of input data

        Returns:
            InferenceInput: Configured input wrapper
        """
        return cls(data)

    @classmethod
    def from_tensor(cls, tensor: Tensor, feature_name: str = "x") -> InferenceInput:
        """Create InferenceInput from single tensor.

        Args:
            tensor: Input tensor
            feature_name: Name to assign to the tensor

        Returns:
            InferenceInput: Configured input wrapper
        """
        return cls({feature_name: tensor})

    @classmethod
    def from_array(cls, array: np.ndarray, feature_name: str = "x") -> InferenceInput:
        """Create InferenceInput from NumPy array.

        Args:
            array: Input array
            feature_name: Name to assign to the array

        Returns:
            InferenceInput: Configured input wrapper
        """
        return cls({feature_name: array})

    @classmethod
    def from_file(cls, file_path: Path | str) -> InferenceInput:
        """Create InferenceInput from file path.

        Args:
            file_path: Path to input data file

        Returns:
            InferenceInput: Configured input wrapper
        """
        return cls(file_path)

    def __repr__(self) -> str:
        """String representation of the inference input."""
        return f"InferenceInput(adapter={self.get_adapter_type()}, features={self.get_feature_names()})"