"""Input adapters for converting various input formats to tensor dictionaries.

This module provides adapter classes that handle the conversion of different
input formats to the standardized tensor dictionary format used by DLKit models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import numpy as np
from torch import Tensor


class InputAdapter(ABC):
    """Abstract base class for input adapters.

    Input adapters convert various input formats to the standardized
    tensor dictionary format expected by DLKit models.
    """

    @abstractmethod
    def convert_to_tensor_dict(
        self,
        data: Any,
        feature_names: list[str] | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32
    ) -> dict[str, Tensor]:
        """Convert input data to tensor dictionary.

        Args:
            data: Input data in adapter-specific format
            feature_names: Expected feature names
            device: Target device for tensors
            dtype: Target dtype for tensors

        Returns:
            Dictionary mapping feature names to tensors
        """
        pass


class TensorInputAdapter(InputAdapter):
    """Adapter for single torch.Tensor inputs."""

    def convert_to_tensor_dict(
        self,
        data: Tensor,
        feature_names: list[str] | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32
    ) -> dict[str, Tensor]:
        """Convert single tensor to tensor dictionary.

        Args:
            data: Input tensor
            feature_names: Feature names (uses first name or "x" as default)
            device: Target device for tensor
            dtype: Target dtype for tensor

        Returns:
            Dictionary with single tensor entry
        """
        # Determine feature name
        if feature_names and len(feature_names) > 0:
            feature_name = feature_names[0]
        else:
            feature_name = "x"

        # Convert to target device and dtype
        tensor = data.to(device=device, dtype=dtype)

        return {feature_name: tensor}


class DictInputAdapter(InputAdapter):
    """Adapter for dictionary inputs (tensors, arrays, or mixed)."""

    def convert_to_tensor_dict(
        self,
        data: dict[str, Any] | list[dict[str, Any]],
        feature_names: list[str] | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32
    ) -> dict[str, Tensor]:
        """Convert dictionary or list of dictionaries to tensor dictionary.

        Args:
            data: Dictionary of data or list of dictionaries
            feature_names: Expected feature names (for validation)
            device: Target device for tensors
            dtype: Target dtype for tensors

        Returns:
            Dictionary mapping feature names to tensors

        Raises:
            ValueError: If data format is invalid
        """
        if isinstance(data, list):
            # Handle batch of dictionaries
            return self._convert_batch_to_tensor_dict(data, device, dtype)
        elif isinstance(data, dict):
            # Handle single dictionary
            return self._convert_single_dict_to_tensor_dict(data, device, dtype)
        else:
            raise ValueError(f"Expected dict or list[dict], got {type(data)}")

    def _convert_single_dict_to_tensor_dict(
        self,
        data: dict[str, Any],
        device: torch.device | str,
        dtype: torch.dtype
    ) -> dict[str, Tensor]:
        """Convert single dictionary to tensor dictionary."""
        tensor_dict = {}

        for key, value in data.items():
            tensor_dict[key] = self._convert_value_to_tensor(value, device, dtype)

        return tensor_dict

    def _convert_batch_to_tensor_dict(
        self,
        data: list[dict[str, Any]],
        device: torch.device | str,
        dtype: torch.dtype
    ) -> dict[str, Tensor]:
        """Convert list of dictionaries to batched tensor dictionary."""
        if not data:
            return {}

        # Get feature names from first item
        feature_names = list(data[0].keys())

        # Validate all items have same keys
        for i, item in enumerate(data):
            if set(item.keys()) != set(feature_names):
                raise ValueError(
                    f"Inconsistent keys in batch item {i}: "
                    f"expected {feature_names}, got {list(item.keys())}"
                )

        # Convert and stack tensors for each feature
        tensor_dict = {}
        for feature_name in feature_names:
            # Extract values for this feature from all items
            feature_values = [item[feature_name] for item in data]

            # Convert each value to tensor
            tensors = [
                self._convert_value_to_tensor(value, device, dtype)
                for value in feature_values
            ]

            # Stack into single tensor
            try:
                tensor_dict[feature_name] = torch.stack(tensors, dim=0)
            except Exception as e:
                raise ValueError(
                    f"Failed to stack tensors for feature '{feature_name}': {e}"
                ) from e

        return tensor_dict

    def _convert_value_to_tensor(
        self,
        value: Any,
        device: torch.device | str,
        dtype: torch.dtype
    ) -> Tensor:
        """Convert a single value to tensor."""
        if isinstance(value, Tensor):
            return value.to(device=device, dtype=dtype)
        elif isinstance(value, np.ndarray):
            return torch.from_numpy(value).to(device=device, dtype=dtype)
        elif isinstance(value, (int, float)):
            return torch.tensor(value, device=device, dtype=dtype)
        elif isinstance(value, (list, tuple)):
            return torch.tensor(value, device=device, dtype=dtype)
        else:
            raise ValueError(f"Cannot convert value of type {type(value)} to tensor")


class ArrayInputAdapter(InputAdapter):
    """Adapter for NumPy array inputs."""

    def convert_to_tensor_dict(
        self,
        data: np.ndarray,
        feature_names: list[str] | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32
    ) -> dict[str, Tensor]:
        """Convert NumPy array to tensor dictionary.

        Args:
            data: Input NumPy array
            feature_names: Feature names (uses first name or "x" as default)
            device: Target device for tensor
            dtype: Target dtype for tensor

        Returns:
            Dictionary with single tensor entry
        """
        # Determine feature name
        if feature_names and len(feature_names) > 0:
            feature_name = feature_names[0]
        else:
            feature_name = "x"

        # Convert to tensor
        tensor = torch.from_numpy(data).to(device=device, dtype=dtype)

        return {feature_name: tensor}


class FileInputAdapter(InputAdapter):
    """Adapter for file path inputs."""

    def convert_to_tensor_dict(
        self,
        data: Path | str,
        feature_names: list[str] | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32
    ) -> dict[str, Tensor]:
        """Convert file path to tensor dictionary.

        Args:
            data: File path to load
            feature_names: Expected feature names
            device: Target device for tensors
            dtype: Target dtype for tensors

        Returns:
            Dictionary mapping feature names to tensors

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported

        Note:
            Currently supports .pt/.pth (torch), .npy (numpy), and .npz (numpy) files.
        """
        file_path = Path(data)

        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        # Determine loading method based on file extension
        suffix = file_path.suffix.lower()

        if suffix in [".pt", ".pth"]:
            return self._load_torch_file(file_path, feature_names, device, dtype)
        elif suffix == ".npy":
            return self._load_numpy_file(file_path, feature_names, device, dtype)
        elif suffix == ".npz":
            return self._load_numpy_archive(file_path, feature_names, device, dtype)
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Supported formats: .pt, .pth, .npy, .npz"
            )

    def _load_torch_file(
        self,
        file_path: Path,
        feature_names: list[str] | None,
        device: torch.device | str,
        dtype: torch.dtype
    ) -> dict[str, Tensor]:
        """Load PyTorch tensor file."""
        try:
            data = torch.load(file_path, map_location=device, weights_only=False)
        except Exception as e:
            raise ValueError(f"Failed to load PyTorch file {file_path}: {e}") from e

        if isinstance(data, Tensor):
            # Single tensor file
            feature_name = feature_names[0] if feature_names else "x"
            return {feature_name: data.to(dtype=dtype)}
        elif isinstance(data, dict):
            # Dictionary of tensors
            tensor_dict = {}
            for key, value in data.items():
                if isinstance(value, Tensor):
                    tensor_dict[key] = value.to(device=device, dtype=dtype)
                else:
                    tensor_dict[key] = torch.tensor(value, device=device, dtype=dtype)
            return tensor_dict
        else:
            raise ValueError(f"Unsupported data format in PyTorch file: {type(data)}")

    def _load_numpy_file(
        self,
        file_path: Path,
        feature_names: list[str] | None,
        device: torch.device | str,
        dtype: torch.dtype
    ) -> dict[str, Tensor]:
        """Load NumPy array file."""
        try:
            array = np.load(file_path)
        except Exception as e:
            raise ValueError(f"Failed to load NumPy file {file_path}: {e}") from e

        feature_name = feature_names[0] if feature_names else "x"
        tensor = torch.from_numpy(array).to(device=device, dtype=dtype)

        return {feature_name: tensor}

    def _load_numpy_archive(
        self,
        file_path: Path,
        feature_names: list[str] | None,
        device: torch.device | str,
        dtype: torch.dtype
    ) -> dict[str, Tensor]:
        """Load NumPy archive file (.npz)."""
        try:
            archive = np.load(file_path)
        except Exception as e:
            raise ValueError(f"Failed to load NumPy archive {file_path}: {e}") from e

        tensor_dict = {}
        for key in archive.files:
            array = archive[key]
            tensor = torch.from_numpy(array).to(device=device, dtype=dtype)
            tensor_dict[key] = tensor

        return tensor_dict