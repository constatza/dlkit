"""Dataset and datamodule construction helpers for workflow strategies."""

from __future__ import annotations

from collections.abc import Sized
from pathlib import Path
from typing import Any, cast

from lightning.pytorch import LightningDataModule

from dlkit.engine.workflows.selectors.family_selector import DatasetFamilySelector
from dlkit.infrastructure.config.core.context import BuildContext
from dlkit.infrastructure.config.core.factories import FactoryProvider
from dlkit.infrastructure.config.data_entries import (
    DataEntry,
    FeatureType,
    TargetType,
)
from dlkit.infrastructure.config.enums import DatasetFamily
from dlkit.infrastructure.io.split_provider import get_or_create_split
from dlkit.infrastructure.io.tensor_entries import convert_totensor_entries
from dlkit.infrastructure.types.split import IndexSplit

from .module_defaults import with_runtime_module_defaults


class DatasetBuilder:
    """Build datasets, splits, and datamodules for runtime workflows."""

    def build_context(self, settings: Any) -> BuildContext:
        """Create the shared build context for a workflow."""
        mode = (
            "inference"
            if (settings.SESSION and getattr(settings.SESSION, "inference", False))
            else "training"
        )
        try:
            from dlkit.infrastructure.io.locations import root as root_path

            working_directory = root_path()
        except Exception:
            working_directory = Path.cwd()
        return BuildContext(mode=mode, working_directory=working_directory)

    def build_flexible_dataset(
        self,
        settings: Any,
        context: BuildContext,
        selected_features: tuple[DataEntry, ...],
        selected_targets: tuple[DataEntry, ...],
    ) -> object:
        """Build a flexible dataset from explicit feature and target entries."""
        ds_settings = with_runtime_module_defaults(settings.DATASET)
        if ds_settings is None:
            raise ValueError("DATASET settings are required but not configured")

        ds_name = str(getattr(ds_settings, "name", "")).lower()
        if "supervisedarraydataset" in ds_name:
            from dlkit.engine.data.datasets.flexible import FlexibleDataset

            return FlexibleDataset(
                features=cast(tuple[FeatureType, ...], selected_features),
                targets=cast(tuple[TargetType, ...], selected_targets),
                memmap_cache_dir=getattr(ds_settings, "resolved_memmap_cache_dir", None),
            )

        ds_overrides = {
            "features": selected_features,
            "targets": selected_targets,
        }
        return FactoryProvider.create_component(ds_settings, context.with_overrides(**ds_overrides))

    def build_dataset_with_tensor_entries(
        self,
        settings: Any,
        context: BuildContext,
    ) -> object:
        """Build graph/timeseries datasets after resolving entry tensors eagerly."""
        ds_overrides: dict[str, Any] = {}
        ds_settings = settings.DATASET
        if ds_settings is not None:
            features_path = getattr(ds_settings, "features_path", None)
            resolved_features = convert_totensor_entries(getattr(ds_settings, "features", ()) or ())
            resolved_targets = convert_totensor_entries(getattr(ds_settings, "targets", ()) or ())
            if resolved_features:
                ds_overrides["features"] = resolved_features
            elif features_path is not None:
                ds_overrides["features"] = features_path
            if resolved_targets:
                ds_overrides["targets"] = resolved_targets

        dataset_settings = with_runtime_module_defaults(settings.DATASET)
        if dataset_settings is None:
            raise ValueError("DATASET settings are required but not configured")
        return FactoryProvider.create_component(
            dataset_settings,
            context.with_overrides(**ds_overrides),
        )

    def build_split(self, settings: Any, dataset: object) -> IndexSplit:
        """Get or create the dataset split."""
        if settings.DATASET is None:
            raise ValueError("DATASET settings are required but not configured")
        split_cfg = settings.DATASET.split
        return get_or_create_split(
            num_samples=len(cast(Sized, dataset)),
            test_ratio=split_cfg.test_ratio,
            val_ratio=split_cfg.val_ratio,
            session_name=settings.SESSION.name,
            explicit_filepath=split_cfg.filepath,
        )

    def build_datamodule(
        self,
        settings: Any,
        context: BuildContext,
        dataset: object,
        index_split: IndexSplit,
        *,
        family: DatasetFamily | None = None,
    ) -> LightningDataModule:
        """Build the configured datamodule with optional family defaults."""
        datamodule_settings = with_runtime_module_defaults(settings.DATAMODULE)
        if datamodule_settings is None:
            raise ValueError("DATAMODULE settings are required but not configured")

        effective_settings = datamodule_settings
        if family is not None:
            name = getattr(effective_settings, "name", None)
            if not name or str(name) == "InMemoryModule":
                datamodule_class = DatasetFamilySelector.default_datamodule_class_for_family(family)
                effective_settings = effective_settings.model_copy(
                    update={"name": datamodule_class}
                )

        dm_context = context.with_overrides(
            dataset=dataset,
            split=index_split,
            dataloader=effective_settings.dataloader,
        )
        return FactoryProvider.create_component(effective_settings, dm_context)
