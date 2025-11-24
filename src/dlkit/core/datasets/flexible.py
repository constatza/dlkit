from pathlib import Path
from typing import Any
from collections.abc import Iterable

import numpy as np
import torch
from torch import Tensor

from dlkit.tools.io import load_array
from .base import BaseDataset, register_dataset


def _normalize_entries(entries: Any) -> dict[str, Path | Tensor | np.ndarray]:
    """Normalize various entry specs into a mapping of name -> (path OR value).

    Supports:
    - list[settings.data_entries.Feature | settings.data_entries.Target] (with .path or .value)
    - list[dict] with keys {"name", "path"}
    - dict[name, path]

    For DataEntry objects (Feature/Target):
    - If entry.has_value() is True, extracts the in-memory .value (tensor/array)
    - Otherwise, extracts the file path from .path attribute
    - This enables both file-based (production) and in-memory (testing) workflows

    Args:
        entries: Collection of entry specifications

    Returns:
        Dictionary mapping entry name to either a file path or in-memory tensor/array
    """
    result: dict[str, Path | Tensor | np.ndarray] = {}
    if entries is None:
        return result

    # dict[name -> path]
    if isinstance(entries, dict):
        for k, v in entries.items():
            result[str(k)] = Path(v)
        return result

    # list[...] entries
    for item in entries:  # type: ignore[assignment]
        # Check for DataEntry objects with .value attribute (in-memory data)
        if hasattr(item, "has_value") and callable(item.has_value) and item.has_value():
            result[str(item.name)] = item.value
        # Check for DataEntry objects with .path attribute (file-based data)
        elif hasattr(item, "has_path") and callable(item.has_path) and item.has_path():
            result[str(item.name)] = Path(item.path)
        # Legacy support: direct name/path attributes
        elif hasattr(item, "name") and hasattr(item, "path"):
            result[str(getattr(item, "name"))] = Path(getattr(item, "path"))
        elif isinstance(item, dict):
            name = item.get("name")
            path = item.get("path")
            if name is None or path is None:
                raise ValueError("Entry dict must contain both 'name' and 'path'")
            result[str(name)] = Path(path)
        else:
            raise TypeError("Unsupported entry type; expected settings entry, dict, or mapping")

    return result


def _load_or_convert_tensor(
    source: Path | Tensor | np.ndarray,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Pure function: convert source to torch.Tensor with dtype handling.

    Handles both file paths (production) and in-memory data (testing).
    Respects precision context via PrecisionService for dtype resolution.

    Args:
        source: File path OR in-memory tensor/array
        dtype: Target dtype (uses PrecisionService if None)

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
    return load_array(source, dtype=dtype)


@register_dataset
class FlexibleDataset(BaseDataset):
    """Dataset that loads an arbitrary set of feature and target files.

    Entries are provided as collections of name/path pairs. The key used in
    __getitem__ output is the entry name, and the value is the tensor slice
    at the requested index.

    Precision handling is automatic via the global precision service. Use
    precision_override() context to control the dtype of loaded tensors.
    """

    def __init__(
        self,
        *,
        features: Iterable[Any] | dict[str, Any],
        targets: Iterable[Any] | dict[str, Any] | None = None,
    ) -> None:
        feat_map = _normalize_entries(features)
        targ_map = _normalize_entries(targets)

        if not feat_map and not targ_map:
            raise ValueError("At least one feature or target entry is required")

        # Precision is automatically resolved from global precision service
        # which checks precision context (set via precision_override())
        # Handles both file paths (production) and in-memory values (testing)
        self.features: dict[str, Tensor] = {
            k: _load_or_convert_tensor(v) for k, v in feat_map.items()
        }
        self.targets: dict[str, Tensor] = {
            k: _load_or_convert_tensor(v) for k, v in targ_map.items()
        }

        # Validate consistent length across all tensors
        all_tensors = list(self.features.values()) + list(self.targets.values())
        lengths = {int(t.size(0)) for t in all_tensors}
        if len(lengths) > 1:
            raise ValueError("All feature/target arrays must share the same first dimension")

        self._length = next(iter(lengths)) if lengths else 0

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        for k, t in self.features.items():
            out[k] = t[idx]
        for k, t in self.targets.items():
            out[k] = t[idx]
        return out

