"""Flexible dataset implementation for arbitrary feature/target configurations.

This module provides FlexibleDataset that loads an arbitrary set of feature
and target files based on data entry configurations.
"""

from pathlib import Path
from typing import Any
from collections.abc import Iterable

import numpy as np
import torch
from torch import Tensor

from dlkit.tools.io import load_array
from dlkit.tools.config.data_entries import (
    DataEntry,
    FeatureType,
    TargetType,
    PathBasedEntry,
    ValueBasedEntry,
    TensorDataEntry,
    to_tensor_entry,
    is_feature_entry,
    is_target_entry,
)
from .base import BaseDataset, register_dataset


class PlaceholderNotResolvedError(ValueError):
    """Raised when a placeholder entry is used without value injection."""

    def __init__(self, entry_name: str) -> None:
        """Initialize with entry name.

        Args:
            entry_name: Name of the unresolved placeholder entry
        """
        super().__init__(
            f"Entry '{entry_name}' is a placeholder without path or value. "
            f"Either specify 'path' in config or inject 'value' programmatically."
        )


def _normalize_entries(entries: Any) -> dict[str, tuple[Path | Tensor | np.ndarray, str]]:
    """Extract path or value from DataEntry objects or pre-resolved tensor entries.

    Expects DataEntry objects (PathFeature, ValueFeature, PathTarget, ValueTarget)
    created by Feature() or Target() factories. These factories handle validation.

    Single Responsibility: Extract data sources from validated entries.
    No validation - trust that factories already validated.

    Args:
        entries: Collection of DataEntry objects

    Returns:
        Dictionary mapping entry name to tuple of (data source, entry name).
        The entry name is used as array_key when loading .npz files.

    Raises:
        TypeError: If receives dict (should use Feature()/Target() instead)
        PlaceholderNotResolvedError: If entry is placeholder without data
    """
    result: dict[str, tuple[Path | Tensor | np.ndarray, str]] = {}
    if entries is None:
        return result

    # Reject dicts - force users to use factories
    if isinstance(entries, dict):
        raise TypeError(
            "FlexibleDataset no longer accepts raw dicts. "
            "Use Feature() or Target() factories instead:\n"
            "  from dlkit.tools.config.data_entries import Feature, Target\n"
            "  features = [Feature(name='x', path='data.npy')]"
        )

    # list[DataEntry] entries
    for item in entries:
        # Already-resolved tensor entries
        if isinstance(item, TensorDataEntry):
            result[item.name] = (item.tensor, item.name)
            continue

        # Reject dicts in list
        if isinstance(item, dict):
            raise TypeError(
                "FlexibleDataset no longer accepts raw dicts. "
                "Use Feature(**dict) or Target(**dict) factories instead."
            )

        # ValueBasedEntry: extract in-memory value
        if isinstance(item, ValueBasedEntry):
            if item.is_placeholder():
                raise PlaceholderNotResolvedError(str(item.name or "unknown"))
            assert item.name is not None, "Non-placeholder entry must have name"
            result[item.name] = (item.value, item.name)  # type: ignore[assignment]

        # PathBasedEntry: extract file path
        elif isinstance(item, PathBasedEntry):
            if item.is_placeholder():
                raise PlaceholderNotResolvedError(str(item.name or "unknown"))
            assert item.name is not None, "Non-placeholder entry must have name"
            result[item.name] = (Path(item.path), item.name)  # type: ignore[arg-type]

        # Generic DataEntry: check capabilities
        elif isinstance(item, DataEntry):
            if item.is_placeholder():
                raise PlaceholderNotResolvedError(str(item.name or "unknown"))
            assert item.name is not None, "Non-placeholder entry must have name"
            tensor_entry = to_tensor_entry(item)
            result[item.name] = (tensor_entry.tensor, item.name)

        else:
            raise TypeError(
                f"Unsupported entry type: {type(item).__name__}. "
                f"Expected DataEntry objects from Feature()/Target() factories."
            )

    return result


def _load_or_convert_tensor(
    source: Path | Tensor | np.ndarray,
    dtype: torch.dtype | None = None,
    array_key: str | None = None,
) -> Tensor:
    """Pure function: convert source to torch.Tensor with dtype handling.

    Handles both file paths (production) and in-memory data (testing).
    Respects precision context via PrecisionService for dtype resolution.

    Args:
        source: File path OR in-memory tensor/array
        dtype: Target dtype (uses PrecisionService if None)
        array_key: For .npz files, the array name to extract

    Returns:
        torch.Tensor with appropriate dtype
    """
    # Case 1: Already a torch.Tensor or numpy array (in-memory data)
    if isinstance(source, (torch.Tensor, np.ndarray)):
        tensor = torch.as_tensor(source)  # Zero-copy for numpy arrays

        # Apply dtype conversion if specified
        if dtype is not None:
            return tensor.to(dtype=dtype)

        # Use PrecisionService for dtype if not specified
        # This respects the global precision context set by precision_override()
        from dlkit.interfaces.api.services.precision_service import get_precision_service

        precision_service = get_precision_service()
        resolved_dtype = precision_service.get_torch_dtype()
        return tensor.to(dtype=resolved_dtype)

    # Case 2: File path - delegate to existing load_array()
    # load_array() already handles PrecisionService integration
    # For .npz files, pass array_key to select specific array
    if isinstance(source, Path) and source.suffix.lower() == ".npz":
        return load_array(source, dtype=dtype, array_key=array_key)
    return load_array(source, dtype=dtype)


@register_dataset
class FlexibleDataset(BaseDataset):
    """Dataset that loads an arbitrary set of feature and target files.

    Entries are provided as DataEntry objects created via Feature() or Target()
    factories. The key used in __getitem__ output is the entry name, and the
    value is the tensor slice at the requested index.

    Precision handling is automatic via the global precision service. Use
    precision_override() context to control the dtype of loaded tensors.

    Supported File Formats:
    - NumPy arrays: .npy (single array), .npz (multi-array)
    - PyTorch tensors: .pt, .pth
    - Text files: .txt, .csv

    For .npz files with multiple arrays, the entry name is used as the array key
    to select which array to load from the file.

    Supports:
    - Path-based entries: Data loaded from files (PathFeature, PathTarget)
    - Value-based entries: In-memory data (ValueFeature, ValueTarget)
    - Placeholder entries: Must be resolved before use (raises PlaceholderNotResolvedError)

    Single Responsibility: Load and manage dataset lifecycle (NO validation).
    Validation is handled by Feature()/Target() factories.

    Examples:
        Basic usage with .npy files:
            >>> from dlkit.tools.config.data_entries import Feature, Target
            >>> features = [Feature(name="x", path="data.npy")]
            >>> targets = [Target(name="y", path="labels.npy")]
            >>> dataset = FlexibleDataset(features=features, targets=targets)

        Using .npz files (entry name used as array key):
            >>> features = [Feature(name="features", path="data.npz")]
            >>> targets = [Target(name="targets", path="data.npz")]
            >>> dataset = FlexibleDataset(features=features, targets=targets)
            # Loads array "features" and "targets" from data.npz

        Multiple features from same .npz file:
            >>> features = [
            ...     Feature(name="features", path="data.npz"),
            ...     Feature(name="latent", path="data.npz")
            ... ]
            >>> dataset = FlexibleDataset(features=features)

        Mixed file formats:
            >>> features = [
            ...     Feature(name="x", path="features.npy"),
            ...     Feature(name="y", path="extra.npz")  # Uses "y" as array key
            ... ]
            >>> dataset = FlexibleDataset(features=features)
    """

    def __init__(
        self,
        *,
        features: Iterable[FeatureType],
        targets: Iterable[TargetType] | None = None,
    ) -> None:
        """Initialize FlexibleDataset with feature and target entries.

        Args:
            features: Feature entries (PathFeature or ValueFeature from Feature() factory)
            targets: Target entries (PathTarget or ValueTarget from Target() factory)

        Raises:
            ValueError: If no features or targets are provided
            PlaceholderNotResolvedError: If placeholder entry without value
            TypeError: If raw dicts are passed (use Feature()/Target() instead)
        """
        feat_map = _normalize_entries(features)
        targ_map = _normalize_entries(targets)

        if not feat_map and not targ_map:
            raise ValueError("At least one feature or target entry is required")

        # Precision is automatically resolved from global precision service
        # which checks precision context (set via precision_override())
        # Handles both file paths (production) and in-memory values (testing)
        # For .npz files, uses entry name as array_key
        self.features: dict[str, Tensor] = {
            k: _load_or_convert_tensor(source, array_key=name)
            for k, (source, name) in feat_map.items()
        }
        self.targets: dict[str, Tensor] = {
            k: _load_or_convert_tensor(source, array_key=name)
            for k, (source, name) in targ_map.items()
        }

        # Track entry lengths (no broadcasting of unit-length/constants)
        self._entry_lengths: dict[str, int] = {}
        scalar_entries: set[str] = set()
        non_scalar_lengths: set[int] = set()
        for name, tensor in {**self.features, **self.targets}.items():
            if tensor.dim() == 0:
                self._entry_lengths[name] = 1
                scalar_entries.add(name)
                continue

            length = int(tensor.size(0))
            if length < 1:
                raise ValueError("Feature/target tensors must have at least one sample")

            self._entry_lengths[name] = length
            non_scalar_lengths.add(length)

        if non_scalar_lengths:
            if len(non_scalar_lengths) > 1:
                raise ValueError("Feature/target arrays must share the same first dimension")
            self._length = non_scalar_lengths.pop()
            if scalar_entries and self._length > 1:
                raise ValueError(
                    "Scalar feature/target entries cannot be broadcast; provide per-sample values "
                    "or remove scalars when dataset length exceeds one."
                )
        else:
            self._length = 1

        if len(self._entry_lengths) == 0:
            raise ValueError(
                "At least one feature or target entry is required after validation"
            )

    def __len__(self) -> int:
        """Return number of samples in dataset.

        Returns:
            Number of samples (first dimension of tensors)
        """
        return self._length

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get sample at index.

        Args:
            idx: Sample index

        Returns:
            Dictionary mapping entry names to tensor slices
        """
        out: dict[str, Tensor] = {}
        for k, t in self.features.items():
            if t.dim() == 0:
                out[k] = t  # scalar constant
            else:
                out[k] = t[idx]
        for k, t in self.targets.items():
            if t.dim() == 0:
                out[k] = t  # scalar constant
            else:
                out[k] = t[idx]
        return out
