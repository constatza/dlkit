"""Shared DataModuleSelector resolution via FactoryProvider.create_component()."""

from __future__ import annotations

from typing import cast

from lightning.pytorch import LightningDataModule

from dlkit.engine.workflows.selectors.family_selector import DatasetFamilySelector
from dlkit.infrastructure.config.core.context import BuildContext
from dlkit.infrastructure.config.core.factories import FactoryProvider
from dlkit.infrastructure.config.data_settings import DataSettings
from dlkit.infrastructure.config.dataloader_settings import DataloaderSettings
from dlkit.infrastructure.config.enums import DatasetFamily
from dlkit.infrastructure.io.split_provider import SplitResolution

_DATAMODULE_MODULE = "dlkit.engine.adapters.lightning.datamodules"


def build_datamodule_from_selector(
    data: DataSettings,
    dataset: object,
    split_resolution: SplitResolution,
    context: BuildContext,
    *,
    family: DatasetFamily | None = None,
) -> LightningDataModule:
    """Resolve and construct the LightningDataModule described by data.module.

    Routes through ``FactoryProvider.create_component()`` — the same
    mechanism every other component uses — with ``dataset``/``split``/
    ``dataloader`` passed via ``BuildContext.overrides``, shared by both the
    training and inference datamodule-building code paths. When ``family``
    implies a non-default datamodule class and the selector still holds its
    default name, the resolved class is substituted into the selector before
    construction, since ``DefaultComponentFactory`` accepts an
    already-resolved class object for ``name`` directly.

    Args:
        data: Unified data settings (dataset + dataloader + module selector).
        dataset: The constructed dataset object.
        split_resolution: Resolved train/val/test split.
        context: Shared build context; ``dataset``/``split``/``dataloader``
            are injected via ``context.with_overrides(...)``.
        family: Optional DatasetFamily for datamodule class defaults (e.g. graph).

    Returns:
        Configured LightningDataModule.
    """
    selector = data.module
    selector_updates: dict[str, object] = {}
    if family is not None and selector.name == "ArrayDataModule":
        selector_updates["name"] = DatasetFamilySelector.default_datamodule_class_for_family(family)
    if selector.module_path is None:
        selector_updates["module_path"] = _DATAMODULE_MODULE
    if selector_updates:
        selector = selector.model_copy(update=selector_updates)

    dataloader = DataloaderSettings(
        batch_size=data.batch_size,
        num_workers=data.num_workers,
        shuffle=data.shuffle,
        pin_memory=data.pin_memory,
        persistent_workers=data.persistent_workers,
        prefetch_factor=data.prefetch_factor,
        follow_batch=data.follow_batch,
    )
    built = FactoryProvider.create_component(
        selector,
        context.with_overrides(
            dataset=dataset,
            split=split_resolution.index_split,
            dataloader=dataloader,
        ),
    )
    return cast(LightningDataModule, built)
