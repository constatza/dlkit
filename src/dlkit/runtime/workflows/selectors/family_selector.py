"""Dataset-family selector and default component lookup."""

from __future__ import annotations

from typing import Any

from dlkit.runtime.data.families import resolve_family, resolve_family_from_dataset
from dlkit.tools.config.enums import DatasetFamily


class DatasetFamilySelector:
    """Resolve dataset family and provide default components.

    This small selector centralizes family detection and default choices
    to keep strategies lean and SOLID.
    """

    @staticmethod
    def is_graph(settings: Any) -> bool:
        return resolve_family(settings) is DatasetFamily.GRAPH

    @staticmethod
    def is_timeseries(settings: Any) -> bool:
        return resolve_family(settings) is DatasetFamily.TIMESERIES

    @staticmethod
    def default_datamodule_class_for(settings: Any) -> Any:
        """Get default datamodule class for the given settings.

        Args:
            settings: Workflow configuration settings.

        Returns:
            Default datamodule class for the detected dataset family.
        """
        family = resolve_family(settings)
        return DatasetFamilySelector.default_datamodule_class_for_family(family)

    @staticmethod
    def default_wrapper_class_for(settings: Any) -> Any:
        """Get default wrapper class for the given settings.

        Args:
            settings: Workflow configuration settings.

        Returns:
            Default wrapper class for the detected dataset family.
        """
        family = resolve_family(settings)
        return DatasetFamilySelector.default_wrapper_class_for_family(family)

    @staticmethod
    def resolve_family_from_dataset(dataset: object) -> DatasetFamily:
        """Resolve dataset family from a constructed dataset instance.

        Prefers instance-based detection over settings hints.
        """
        return resolve_family_from_dataset(dataset)

    @staticmethod
    def resolve_family(settings: Any) -> DatasetFamily:
        """Resolve dataset family from settings.

        Args:
            settings: Workflow configuration settings.

        Returns:
            Detected DatasetFamily enum value.
        """
        return resolve_family(settings)

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
                from dlkit.runtime.adapters.lightning.datamodules.graph import GraphDataModule

                return GraphDataModule
            case DatasetFamily.TIMESERIES:
                from dlkit.runtime.adapters.lightning.datamodules.timeseries import (
                    TimeSeriesDataModule,
                )

                return TimeSeriesDataModule
            case _:
                from dlkit.runtime.adapters.lightning.datamodules.array import InMemoryModule

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
                from dlkit.runtime.adapters.lightning.graph import GraphLightningWrapper

                return GraphLightningWrapper
            case DatasetFamily.TIMESERIES:
                from dlkit.runtime.adapters.lightning.timeseries import TimeSeriesLightningWrapper

                return TimeSeriesLightningWrapper
            case _:
                from dlkit.runtime.adapters.lightning.standard import StandardLightningWrapper

                return StandardLightningWrapper
