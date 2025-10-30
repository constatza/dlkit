from pathlib import Path
from typing import Any
from collections.abc import Iterable

import torch
from torch import Tensor

from dlkit.tools.io import load_array
from .base import BaseDataset, register_dataset


def _normalize_entries(entries: Any) -> dict[str, Path]:
    """Normalize various entry specs into a mapping of name -> Path.

    Supports:
    - list[settings.data_entries.Feature | settings.data_entries.Target]
    - list[dict] with keys {"name", "path"}
    - dict[name, path]
    """
    result: dict[str, Path] = {}
    if entries is None:
        return result

    # dict[name -> path]
    if isinstance(entries, dict):
        for k, v in entries.items():
            result[str(k)] = Path(v)
        return result

    # list[...] entries
    for item in entries:  # type: ignore[assignment]
        if hasattr(item, "name") and hasattr(item, "path"):
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
        self.features: dict[str, Tensor] = {
            k: load_array(v) for k, v in feat_map.items()
        }
        self.targets: dict[str, Tensor] = {
            k: load_array(v) for k, v in targ_map.items()
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

