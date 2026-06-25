"""Dataset-family selector and default component lookup."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from dlkit.engine.data.families import resolve_family, resolve_family_from_dataset
from dlkit.infrastructure.config.enums import DatasetFamily
from dlkit.infrastructure.config.job_config import JobConfig

if TYPE_CHECKING:
    from lightning import LightningDataModule

    from dlkit.engine.adapters.lightning.base import CoreLightningWrapper


class DatasetFamilySelector:
    """Resolve dataset family and provide default components.

    This small selector centralizes family detection and default choices
    to keep strategies lean and SOLID.
    """

    @staticmethod
    def is_graph(settings: JobConfig) -> bool:
        return resolve_family(settings) is DatasetFamily.GRAPH

    @staticmethod
    def default_datamodule_class_for(settings: JobConfig) -> type[LightningDataModule]:
        """Get default datamodule class for the given settings.

        Args:
            settings: Workflow configuration settings.

        Returns:
            Default datamodule class for the detected dataset family.
        """
        family = resolve_family(settings)
        return DatasetFamilySelector.default_datamodule_class_for_family(family)

    @staticmethod
    def default_wrapper_class_for(settings: JobConfig) -> type[CoreLightningWrapper]:
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
        return resolve_family_from_dataset(cast(Sequence[object], dataset))

    @staticmethod
    def resolve_family(settings: JobConfig) -> DatasetFamily:
        """Resolve dataset family from settings.

        Args:
            settings: Workflow configuration settings.

        Returns:
            Detected DatasetFamily enum value.
        """
        return resolve_family(settings)

    @staticmethod
    def default_datamodule_class_for_family(family: DatasetFamily) -> type[LightningDataModule]:
        """Get default datamodule class for the given dataset family.

        Args:
            family: Dataset family to get default for.

        Returns:
            Default datamodule class.
        """
        match family:
            case DatasetFamily.GRAPH:
                from dlkit.engine.adapters.lightning.datamodules.graph import GraphDataModule

                return GraphDataModule
            case _:
                from dlkit.engine.adapters.lightning.datamodules.array import ArrayDataModule

                return ArrayDataModule

    @staticmethod
    def default_wrapper_class_for_family(family: DatasetFamily) -> type[CoreLightningWrapper]:
        """Get default wrapper class for the given dataset family.

        Args:
            family: Dataset family to get default for.

        Returns:
            Default wrapper class.
        """
        match family:
            case DatasetFamily.GRAPH:
                from dlkit.engine.adapters.lightning.graph import GraphLightningWrapper

                return GraphLightningWrapper
            case _:
                from dlkit.engine.adapters.lightning.standard import StandardLightningWrapper

                return StandardLightningWrapper
