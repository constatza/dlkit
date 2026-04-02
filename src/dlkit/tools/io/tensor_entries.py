"""Tensor-entry conversion helpers for config-defined data entries."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch

from dlkit.tools.config.data_entries import (
    DataEntry,
    PathBasedEntry,
    TransformSettings,
    ValueBasedEntry,
)
from dlkit.tools.io.arrays import load_array


@dataclass(frozen=True, slots=True, kw_only=True)
class TensorDataEntry:
    """Resolved tensor entry shared by datasets and runtime builders."""

    name: str
    tensor: torch.Tensor
    write: bool = False
    transforms: tuple[TransformSettings, ...] = ()


def _coerce_tensor(value: torch.Tensor | np.ndarray, dtype: torch.dtype | None) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def to_tensor_entry(entry: DataEntry) -> TensorDataEntry:
    """Convert a config entry into a concrete tensor entry."""

    name = entry.name or "anonymous"
    dtype = getattr(entry, "dtype", None)
    write = getattr(entry, "write", False)
    transforms = tuple(getattr(entry, "transforms", ()) or ())

    if isinstance(entry, PathBasedEntry):
        if not entry.has_path() or entry.path is None:
            raise ValueError(f"Entry '{name}' is a placeholder without a path or value")
        tensor = load_array(entry.path, dtype=dtype)
        return TensorDataEntry(name=name, tensor=tensor, write=write, transforms=transforms)

    if isinstance(entry, ValueBasedEntry):
        if not entry.has_value() or entry.value is None:
            raise ValueError(f"Entry '{name}' is a placeholder without a value")
        tensor = _coerce_tensor(entry.value, dtype=dtype)
        return TensorDataEntry(name=name, tensor=tensor, write=write, transforms=transforms)

    raise TypeError(f"Unsupported entry type for tensor conversion: {type(entry)}")


def convert_totensor_entries(entries: Iterable[DataEntry]) -> tuple[TensorDataEntry, ...]:
    """Convert config entries to concrete tensor entries."""

    return tuple(to_tensor_entry(entry) for entry in entries)
