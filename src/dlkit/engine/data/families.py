"""Dataset-family resolution helpers for runtime services."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from dlkit.infrastructure.config.enums import DatasetFamily


def _is_graph_hint(settings: Any) -> bool:
    try:
        ds = settings.DATASET
        dm = settings.DATAMODULE
        name_mod = (
            f"{getattr(ds, 'name', '')} {getattr(ds, 'module_path', '')} "
            f"{getattr(dm, 'name', '')} {getattr(dm, 'module_path', '')}"
        ).lower()
        return any(k in name_mod for k in ("graph", "pyg", "geometric"))
    except Exception:
        return False


def resolve_family_from_dataset(dataset: object) -> DatasetFamily:
    """Resolve dataset family from a constructed dataset instance."""
    try:
        from torch_geometric.data import Data as pyg_data
        from torch_geometric.data import InMemoryDataset as pyg_dataset

        if isinstance(dataset, pyg_dataset):
            return DatasetFamily.GRAPH
        try:
            sample = cast(Sequence[object], dataset)[0]
            if isinstance(sample, pyg_data):
                return DatasetFamily.GRAPH
        except Exception:
            pass
    except Exception:
        pass

    return DatasetFamily.FLEXIBLE


def resolve_family(settings: Any) -> DatasetFamily:
    """Resolve dataset family from workflow settings."""
    try:
        explicit = getattr(settings.DATASET, "family", None)
        if isinstance(explicit, DatasetFamily):
            return explicit
        if explicit is not None:
            match str(explicit).lower():
                case "graph":
                    return DatasetFamily.GRAPH
                case _:
                    return DatasetFamily.FLEXIBLE
    except Exception:
        pass

    try:
        explicit = getattr(settings.DATASET, "type", None)
        if isinstance(explicit, DatasetFamily):
            return explicit
        if explicit is not None:
            match str(explicit).lower():
                case "graph":
                    return DatasetFamily.GRAPH
                case _:
                    return DatasetFamily.FLEXIBLE
    except Exception:
        pass

    if _is_graph_hint(settings):
        return DatasetFamily.GRAPH
    return DatasetFamily.FLEXIBLE
