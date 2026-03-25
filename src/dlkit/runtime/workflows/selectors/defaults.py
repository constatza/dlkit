from __future__ import annotations

from typing import Any

from dlkit.tools.config.enums import DatasetFamily


def _is_graph_hint(settings: Any) -> bool:
    try:
        ds = settings.DATASET
        dm = settings.DATAMODULE
        name_mod = f"{getattr(ds, 'name', '')} {getattr(ds, 'module_path', '')} {getattr(dm, 'name', '')} {getattr(dm, 'module_path', '')}".lower()
        return any(k in name_mod for k in ("graph", "pyg", "geometric"))
    except Exception:
        return False


def _is_timeseries_hint(settings: Any) -> bool:
    try:
        ds = settings.DATASET
        dm = settings.DATAMODULE
        name_mod = f"{getattr(ds, 'name', '')} {getattr(ds, 'module_path', '')} {getattr(dm, 'name', '')} {getattr(dm, 'module_path', '')}".lower()
        return any(k in name_mod for k in ("timeseries", "forecast"))
    except Exception:
        return False


class FamilyDefaults:
    """Resolve dataset family and provide default components.

    This small selector centralizes family detection and default choices
    to keep strategies lean and SOLID.
    """

    @staticmethod
    def is_graph(settings: Any) -> bool:
        try:
            explicit = getattr(settings.DATASET, "type", None)
            if explicit is not None and str(explicit).lower() == "graph":
                return True
        except Exception:
            pass
        return _is_graph_hint(settings)

    @staticmethod
    def is_timeseries(settings: Any) -> bool:
        try:
            explicit = getattr(settings.DATASET, "type", None)
            if explicit is not None and str(explicit).lower() == "timeseries":
                return True
        except Exception:
            pass
        return _is_timeseries_hint(settings)

    @staticmethod
    def default_datamodule_class_for(settings: Any) -> Any:
        """Get default datamodule class for the given settings.

        Args:
            settings: Workflow configuration settings.

        Returns:
            Default datamodule class for the detected dataset family.
        """
        family = FamilyDefaults.resolve_family(settings)
        return FamilyDefaults.default_datamodule_class_for_family(family)

    @staticmethod
    def default_wrapper_class_for(settings: Any) -> Any:
        """Get default wrapper class for the given settings.

        Args:
            settings: Workflow configuration settings.

        Returns:
            Default wrapper class for the detected dataset family.
        """
        family = FamilyDefaults.resolve_family(settings)
        return FamilyDefaults.default_wrapper_class_for_family(family)

    @staticmethod
    def resolve_family_from_dataset(dataset: object) -> DatasetFamily:
        """Resolve dataset family from a constructed dataset instance.

        Prefers instance-based detection over settings hints.
        """
        # Graph detection via PyG types or samples
        try:
            from torch_geometric.data import Data as _PyGData  # type: ignore
            from torch_geometric.data import InMemoryDataset as _PyGDS  # type: ignore

            if isinstance(dataset, _PyGDS):
                return DatasetFamily.GRAPH
            try:
                sample = dataset[0]  # type: ignore[index]
                if isinstance(sample, _PyGData):
                    return DatasetFamily.GRAPH
            except Exception:
                pass
        except Exception:
            pass

        # Timeseries detection via pytorch_forecasting TimeSeriesDataSet
        try:
            from pytorch_forecasting import TimeSeriesDataSet as _TFDS  # type: ignore

            # Our ForecastingDataset exposes `.timeseries`
            ts = getattr(dataset, "timeseries", None)
            if isinstance(ts, _TFDS):
                return DatasetFamily.TIMESERIES
        except Exception:
            pass

        return DatasetFamily.FLEXIBLE

    @staticmethod
    def resolve_family(settings: Any) -> DatasetFamily:
        """Resolve dataset family from settings.

        Args:
            settings: Workflow configuration settings.

        Returns:
            Detected DatasetFamily enum value.
        """
        try:
            explicit = getattr(settings.DATASET, "type", None)
            if isinstance(explicit, DatasetFamily):
                return explicit
            if explicit is not None:
                match str(explicit).lower():
                    case "graph":
                        return DatasetFamily.GRAPH
                    case "timeseries":
                        return DatasetFamily.TIMESERIES
                    case _:
                        return DatasetFamily.FLEXIBLE
        except Exception:
            pass
        if FamilyDefaults.is_graph(settings):
            return DatasetFamily.GRAPH
        if FamilyDefaults.is_timeseries(settings):
            return DatasetFamily.TIMESERIES
        return DatasetFamily.FLEXIBLE

    @staticmethod
    def default_datamodule_class_for_family(family: DatasetFamily) -> Any:
        """Get default datamodule class for the given dataset family.

        Args:
            family: Dataset family to get default for.

        Returns:
            Default datamodule class.
        """
        match family:
            case DatasetFamily.GRAPH:
                from dlkit.core.datamodules.graph import GraphDataModule

                return GraphDataModule
            case DatasetFamily.TIMESERIES:
                from dlkit.core.datamodules.timeseries import TimeSeriesDataModule

                return TimeSeriesDataModule
            case _:
                from dlkit.core.datamodules.array import InMemoryModule

                return InMemoryModule

    @staticmethod
    def default_wrapper_class_for_family(family: DatasetFamily) -> Any:
        """Get default wrapper class for the given dataset family.

        Args:
            family: Dataset family to get default for.

        Returns:
            Default wrapper class.
        """
        match family:
            case DatasetFamily.GRAPH:
                from dlkit.core.models.wrappers.graph import GraphLightningWrapper

                return GraphLightningWrapper
            case DatasetFamily.TIMESERIES:
                from dlkit.core.models.wrappers.timeseries import TimeSeriesLightningWrapper

                return TimeSeriesLightningWrapper
            case _:
                from dlkit.core.models.wrappers.standard import StandardLightningWrapper

                return StandardLightningWrapper
