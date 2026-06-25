"""Dataset-family resolution helpers for runtime services."""

from __future__ import annotations

from collections.abc import Sequence

from dlkit.infrastructure.config.data_settings import DataSettings
from dlkit.infrastructure.config.enums import DatasetFamily
from dlkit.infrastructure.config.job_config import JobConfig


def _get_ds_dm_names(settings: JobConfig) -> str:
    """Extract a combined name/module_path string for graph-hint detection."""
    ds = settings.data
    if ds is None:
        return ""
    module_name = ds.module.name if ds.module else ""
    module_path = ds.module.module_path or ""
    return (f"{ds.name or ''} {ds.module_path or ''} {module_name} {module_path}").lower()


def _is_graph_hint(settings: JobConfig) -> bool:
    try:
        return any(k in _get_ds_dm_names(settings) for k in ("graph", "pyg", "geometric"))
    except Exception:
        return False


def resolve_family_from_dataset(dataset: Sequence[object]) -> DatasetFamily:
    """Resolve dataset family from a constructed dataset instance."""
    try:
        from torch_geometric.data import Data as pyg_data
        from torch_geometric.data import InMemoryDataset as pyg_dataset

        if isinstance(dataset, pyg_dataset):
            return DatasetFamily.GRAPH
        try:
            if isinstance(dataset[0], pyg_data):
                return DatasetFamily.GRAPH
        except Exception:
            pass
    except Exception:
        pass

    return DatasetFamily.FLEXIBLE


def _resolve_family_from_data_settings(ds_settings: DataSettings) -> DatasetFamily | None:
    """Resolve DatasetFamily from typed DataSettings."""
    if isinstance(ds_settings.family, DatasetFamily):
        return ds_settings.family
    if ds_settings.family is not None:
        return (
            DatasetFamily.GRAPH
            if str(ds_settings.family).lower() == "graph"
            else DatasetFamily.FLEXIBLE
        )
    return None


def resolve_family(settings: JobConfig) -> DatasetFamily:
    """Resolve dataset family from workflow settings."""
    if settings.data is not None:
        result = _resolve_family_from_data_settings(settings.data)
        if result is not None:
            return result
    if _is_graph_hint(settings):
        return DatasetFamily.GRAPH
    return DatasetFamily.FLEXIBLE
