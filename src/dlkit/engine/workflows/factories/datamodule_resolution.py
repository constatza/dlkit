"""Shared DataModuleSelector resolution: not a ComponentSettings, so no FactoryProvider."""

from __future__ import annotations

from lightning.pytorch import LightningDataModule

from dlkit.engine.workflows.selectors.family_selector import DatasetFamilySelector
from dlkit.infrastructure.config.data_settings import DataSettings
from dlkit.infrastructure.config.dataloader_settings import DataloaderSettings
from dlkit.infrastructure.config.enums import DatasetFamily
from dlkit.infrastructure.io.split_provider import SplitResolution
from dlkit.infrastructure.utils.general import import_object

_DATAMODULE_MODULE = "dlkit.engine.adapters.lightning.datamodules"


def build_datamodule_from_selector(
    data: DataSettings,
    dataset: object,
    split_resolution: SplitResolution,
    *,
    family: DatasetFamily | None = None,
) -> LightningDataModule:
    """Resolve and construct the LightningDataModule described by data.module.

    DataModuleSelector is a plain settings selector, not a ComponentSettings — it
    has no get_init_kwargs(), so it must never be routed through
    FactoryProvider.create_component(). This hand-resolves the class and
    constructs it directly, shared by both the training and inference
    datamodule-building code paths.

    Args:
        data: Unified data settings (dataset + dataloader + module selector).
        dataset: The constructed dataset object.
        split_resolution: Resolved train/val/test split.
        family: Optional DatasetFamily for datamodule class defaults (e.g. graph).

    Returns:
        Configured LightningDataModule.
    """
    dm_selector = data.module
    module_path = dm_selector.module_path or _DATAMODULE_MODULE
    dm_name = dm_selector.name or "ArrayDataModule"
    dataloader = DataloaderSettings(
        batch_size=data.batch_size,
        num_workers=data.num_workers,
        shuffle=data.shuffle,
        pin_memory=data.pin_memory,
        persistent_workers=data.persistent_workers,
        prefetch_factor=data.prefetch_factor,
        follow_batch=data.follow_batch,
    )
    # Graph family needs a graph-aware datamodule.
    if family is not None and dm_name == "ArrayDataModule":
        dm_cls = DatasetFamilySelector.default_datamodule_class_for_family(family)
    else:
        dm_cls = import_object(dm_name, fallback_module=module_path)
    return dm_cls(
        dataset=dataset,
        split=split_resolution.index_split,
        dataloader=dataloader,
    )
