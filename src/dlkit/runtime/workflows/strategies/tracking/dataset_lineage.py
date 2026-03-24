"""Dataset lineage extraction and structured logging strategies.

This module keeps lineage extraction separate from tracker orchestration:
- Source path extraction is config-driven (settings are the source of truth)
- Structured dataset payload creation is entry-driven (Feature/Target values or paths)
- Dataset-instance fallbacks are handled via dedicated strategies (e.g. tabular `.df`)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
import warnings

import numpy as np
import torch

from dlkit.runtime.workflows.selectors.defaults import FamilyDefaults
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.enums import DatasetFamily
from dlkit.tools.config.data_entries import IPathBased, IValueBased
from dlkit.tools.io import load_array
from dlkit.tools.utils.logging_config import get_logger

logger = get_logger(__name__)


def _append_unique_path(paths: list[str], candidate: Any) -> None:
    """Append candidate to paths if it is a non-empty path-like value."""
    if candidate is None:
        return
    if not isinstance(candidate, (str, Path)):
        return
    value = str(candidate).strip()
    if value and value not in paths:
        paths.append(value)


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


def _collect_entry_sources(settings: GeneralSettings) -> list[str]:
    """Collect source paths from DATASET.features/targets DataEntry objects."""
    sources: list[str] = []
    ds_settings = settings.DATASET
    if ds_settings is None:
        return sources

    for entries_attr in ("features", "targets"):
        entries = getattr(ds_settings, entries_attr, None) or []
        for entry in entries:
            if isinstance(entry, IPathBased):
                _append_unique_path(sources, entry.get_path())

    return sources


class ISourceExtractionStrategy(Protocol):
    """Config-only path extraction strategy."""

    def collect(self, settings: GeneralSettings) -> list[str]:
        """Return ordered unique source paths."""


class GraphConfigSourceStrategy:
    """Extract graph source paths from graph dataset config fields."""

    _FIELDS = ("x", "edge_index", "y")

    def collect(self, settings: GeneralSettings) -> list[str]:
        sources: list[str] = []
        ds_settings = settings.DATASET
        if ds_settings is None:
            return sources
        for field_name in self._FIELDS:
            _append_unique_path(sources, getattr(ds_settings, field_name, None))
        return sources


class TimeSeriesConfigSourceStrategy:
    """Extract timeseries source paths from explicit config path fields."""

    _FIELDS = ("features_path", "features_file", "data_path", "table_path", "file_path", "path")

    def collect(self, settings: GeneralSettings) -> list[str]:
        sources: list[str] = []
        ds_settings = settings.DATASET
        if ds_settings is None:
            return sources
        for field_name in self._FIELDS:
            _append_unique_path(sources, getattr(ds_settings, field_name, None))
        return sources


class CustomConfigSourceStrategy:
    """Extract custom lineage source list from DATASET.source_paths."""

    def collect(self, settings: GeneralSettings) -> list[str]:
        sources: list[str] = []
        ds_settings = settings.DATASET
        if ds_settings is None:
            return sources

        configured = getattr(ds_settings, "source_paths", None)
        if isinstance(configured, (list, tuple, set)):
            for value in configured:
                _append_unique_path(sources, value)
            return sources

        _append_unique_path(sources, configured)
        return sources


class DatasetSourceCollector:
    """Collect canonical dataset sources from settings using family-specific strategies."""

    def __init__(self) -> None:
        self._graph = GraphConfigSourceStrategy()
        self._timeseries = TimeSeriesConfigSourceStrategy()
        self._custom = CustomConfigSourceStrategy()

    def collect(self, settings: GeneralSettings) -> list[str]:
        sources: list[str] = []
        for value in _collect_entry_sources(settings):
            _append_unique_path(sources, value)

        family = FamilyDefaults.resolve_family(settings)
        match family:
            case DatasetFamily.GRAPH:
                extras = self._graph.collect(settings)
            case DatasetFamily.TIMESERIES:
                extras = self._timeseries.collect(settings)
            case _:
                extras = self._custom.collect(settings)

        for value in extras:
            _append_unique_path(sources, value)
        return sources


@dataclass(frozen=True, slots=True, kw_only=True)
class EntryNumpyPayload:
    """Structured payload derived from DATASET Feature/Target entries."""

    features: dict[str, np.ndarray]
    targets: dict[str, np.ndarray] | None


class EntryNumpyPayloadBuilder:
    """Build numpy payload from DataEntry values/paths.

    Priority:
    1) Path-based entries (load from disk)
    2) Value-based entries (in-memory arrays/tensors)
    """

    def build(self, settings: GeneralSettings) -> EntryNumpyPayload | None:
        ds_settings = settings.DATASET
        if ds_settings is None:
            return None

        features = self._materialize_entries(getattr(ds_settings, "features", None) or [])
        if not features:
            return None

        targets = self._materialize_entries(getattr(ds_settings, "targets", None) or [])
        return EntryNumpyPayload(features=features, targets=targets if targets else None)

    def _materialize_entries(self, entries: list[Any]) -> dict[str, np.ndarray]:
        materialized: dict[str, np.ndarray] = {}
        for entry in entries:
            name = getattr(entry, "name", None)
            if not isinstance(name, str) or not name:
                continue

            value = self._resolve_entry_value(entry, name)

            if value is None:
                continue
            materialized[name] = _as_numpy(value)

        return materialized

    def _resolve_entry_value(self, entry: Any, name: str) -> Any | None:
        if isinstance(entry, IPathBased):
            path = entry.get_path()
            if path is None:
                return None
            path_obj = Path(path)
            if path_obj.is_dir():
                return None  # directory-based formats (e.g. sparse packs) cannot be materialized
            if path_obj.suffix.lower() == ".npz":
                return load_array(path_obj, array_key=name)
            return load_array(path_obj)

        if isinstance(entry, IValueBased):
            return entry.get_value()

        return None


class IStructuredDatasetLoggingStrategy(Protocol):
    """Strategy interface for structured MLflow dataset logging."""

    def log(
        self,
        *,
        dataset: Any,
        run_context: Any,
        settings: GeneralSettings,
        dataset_name: str,
        dataset_source: str | None,
        tags: dict[str, str],
    ) -> bool:
        """Try logging dataset with this strategy; return True on success."""


class EntryStructuredDatasetLoggingStrategy:
    """Log dataset from DATASET Feature/Target entries."""

    def __init__(self, payload_builder: EntryNumpyPayloadBuilder | None = None) -> None:
        self._payload_builder = payload_builder or EntryNumpyPayloadBuilder()

    def log(
        self,
        *,
        dataset: Any,
        run_context: Any,
        settings: GeneralSettings,
        dataset_name: str,
        dataset_source: str | None,
        tags: dict[str, str],
    ) -> bool:
        del dataset  # strategy is intentionally settings-driven
        try:
            import mlflow.data
        except Exception as exc:
            logger.warning("MLflow dataset API unavailable for entry-based logging: {}", exc)
            return False

        payload = self._payload_builder.build(settings)
        if payload is None:
            return False

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The specified dataset source can be interpreted in multiple ways:.*",
                    category=UserWarning,
                )
                mlflow_dataset = mlflow.data.from_numpy(
                    features=payload.features,
                    targets=payload.targets,
                    name=dataset_name,
                    source=dataset_source,
                )
                run_context.log_dataset(
                    mlflow_dataset,
                    context="training",
                    tags=tags if tags else None,
                )
            logger.debug("Logged structured entry-based dataset '{}' to MLflow", dataset_name)
            return True
        except Exception as exc:
            logger.warning("Failed to log entry-based dataset to MLflow: {}", exc)
            return False


class TabularStructuredDatasetLoggingStrategy:
    """Fallback strategy for datasets exposing a tabular `df` attribute."""

    def log(
        self,
        *,
        dataset: Any,
        run_context: Any,
        settings: GeneralSettings,
        dataset_name: str,
        dataset_source: str | None,
        tags: dict[str, str],
    ) -> bool:
        del settings
        try:
            import mlflow.data
        except Exception as exc:
            logger.warning("MLflow dataset API unavailable for tabular logging: {}", exc)
            return False

        dataframe = getattr(dataset, "df", None)
        if dataframe is None:
            return False

        try:
            if hasattr(dataframe, "to_pandas"):
                dataframe = dataframe.to_pandas()
            if dataframe is None:
                return False

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The specified dataset source can be interpreted in multiple ways:.*",
                    category=UserWarning,
                )
                mlflow_dataset = mlflow.data.from_pandas(
                    dataframe,
                    name=dataset_name,
                    source=dataset_source,
                )
                run_context.log_dataset(
                    mlflow_dataset,
                    context="training",
                    tags=tags if tags else None,
                )
            logger.debug("Logged structured tabular dataset '{}' to MLflow", dataset_name)
            return True
        except Exception as exc:
            logger.warning("Failed to log tabular dataset to MLflow: {}", exc)
            return False


class StructuredDatasetLogger:
    """Chain of structured dataset logging strategies."""

    def __init__(
        self,
        strategies: list[IStructuredDatasetLoggingStrategy] | None = None,
    ) -> None:
        self._strategies = strategies or [
            EntryStructuredDatasetLoggingStrategy(),
            TabularStructuredDatasetLoggingStrategy(),
        ]

    def log(
        self,
        *,
        dataset: Any,
        run_context: Any,
        settings: GeneralSettings,
        dataset_name: str,
        sources: list[str],
        tags: dict[str, str],
    ) -> bool:
        dataset_source = sources[0] if sources else None

        for strategy in self._strategies:
            if strategy.log(
                dataset=dataset,
                run_context=run_context,
                settings=settings,
                dataset_name=dataset_name,
                dataset_source=dataset_source,
                tags=tags,
            ):
                return True

        return False
