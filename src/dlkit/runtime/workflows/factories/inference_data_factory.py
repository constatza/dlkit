"""Pure factory for building inference datamodules without training components."""

from __future__ import annotations

from pathlib import Path

from lightning.pytorch import LightningDataModule

from dlkit.tools.config.workflow_configs import InferenceWorkflowConfig
from dlkit.tools.config.core.context import BuildContext
from dlkit.tools.config.core.factories import FactoryProvider
from dlkit.core.datatypes.split import IndexSplit
from dlkit.tools.io.split_provider import get_or_create_split
from dlkit.core.datasets.flexible import FlexibleDataset


def build_inference_datamodule(settings: InferenceWorkflowConfig) -> LightningDataModule:
    """Build a datamodule for inference batch iteration.

    No training wrapper, no loss, no optimizer. Only SESSION, DATASET, DATAMODULE.
    Pure function: no class, no side effects beyond datamodule construction.

    Args:
        settings: Inference workflow configuration with DATASET and DATAMODULE sections.

    Returns:
        Configured LightningDataModule ready for predict_dataloader iteration.

    Raises:
        ValueError: If DATASET or DATAMODULE sections are not configured.
    """
    if settings.DATASET is None or settings.DATAMODULE is None:
        raise ValueError(
            "DATASET and DATAMODULE sections are required for batch inference. "
            "Set has_batch_inference_config=True before calling build_inference_datamodule()."
        )

    try:
        from dlkit.tools.io.locations import root as _root
        cfg_dir = _root()
    except Exception:
        cfg_dir = Path.cwd()

    context = BuildContext(mode="inference", working_directory=cfg_dir)
    ds_settings = settings.DATASET

    selected_features = tuple(getattr(ds_settings, "features", ()) or ())
    selected_targets = tuple(getattr(ds_settings, "targets", ()) or ())

    dataset = FlexibleDataset(
        features=selected_features,
        targets=selected_targets,
        memmap_cache_dir=getattr(ds_settings, "resolved_memmap_cache_dir", None),
    )

    split_cfg = ds_settings.split
    index_split: IndexSplit = get_or_create_split(
        num_samples=len(dataset),
        test_ratio=split_cfg.test_ratio,
        val_ratio=split_cfg.val_ratio,
        session_name=settings.SESSION.name,
        explicit_filepath=split_cfg.filepath,
    )

    dm_context = context.with_overrides(
        dataset=dataset,
        split=index_split,
        dataloader=settings.DATAMODULE.dataloader,
    )
    return FactoryProvider.create_component(settings.DATAMODULE, dm_context)
