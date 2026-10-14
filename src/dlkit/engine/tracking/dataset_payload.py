"""In-memory numpy payload for structured MLflow dataset logging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from dlkit.infrastructure.config.entry_protocols import IValueBased
from dlkit.infrastructure.config.job_config import JobConfig

from .dataset_sources import path_entries

type _WorkflowSettings = JobConfig


def _as_numpy(value: Any) -> np.ndarray:
    """Convert value to numpy array for mlflow.data.from_numpy()."""
    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    elif hasattr(value, "to_numpy"):
        arr = np.asarray(value.to_numpy())
    elif hasattr(value, "numpy"):
        arr = np.asarray(value.numpy())
    else:
        arr = np.asarray(value)

    # Keep scalar entries compatible with map-style dataset logging.
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr


@dataclass(frozen=True, slots=True, kw_only=True)
class EntryNumpyPayload:
    """Structured payload derived from DATASET Feature/Target entries."""

    features: dict[str, np.ndarray]
    targets: dict[str, np.ndarray] | None


class EntryNumpyPayloadBuilder:
    """Build a numpy payload from already-in-memory Feature/Target entries.

    Deliberately never reads a path-based entry's file content: doing so just to log
    lineage would force an unconditional, unbounded-size disk read regardless of format.
    Any dataset with at least one path-based feature/target defers entirely to
    ``PointerDatasetLoggingStrategy`` (pointer-only lineage), rather than emitting a numpy
    record with some features silently missing.
    """

    def build(self, settings: _WorkflowSettings) -> EntryNumpyPayload | None:
        ds_settings = settings.data
        if ds_settings is None or path_entries(settings):
            return None

        features = self._materialize(getattr(ds_settings, "features", None) or ())
        if not features:
            return None

        targets = self._materialize(getattr(ds_settings, "targets", None) or ())
        return EntryNumpyPayload(features=features, targets=targets or None)

    @staticmethod
    def _materialize(entries: Any) -> dict[str, np.ndarray]:
        materialized: dict[str, np.ndarray] = {}
        for entry in entries:
            if not isinstance(entry, IValueBased) or not entry.name:
                continue
            value = entry.get_value()
            if value is None:
                continue
            materialized[entry.name] = _as_numpy(value)
        return materialized
