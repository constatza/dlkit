"""Pure factory for building inference datamodules without training components."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from lightning.pytorch import LightningDataModule

from dlkit.engine.data.datasets.flexible import FlexibleDataset
from dlkit.infrastructure.config.core.base_settings import ComponentSettings
from dlkit.infrastructure.config.core.context import BuildContext
from dlkit.infrastructure.config.core.factories import FactoryProvider
from dlkit.infrastructure.config.job_config import InferenceJobConfig, JobConfig
from dlkit.infrastructure.io.split_provider import get_or_create_split

from .module_defaults import with_runtime_module_defaults


def build_inference_datamodule(
    settings: InferenceJobConfig | object,
) -> LightningDataModule:
    """Build a datamodule for inference batch iteration.

    No training wrapper, no loss, no optimizer. Only run/experiment, data sections.
    Pure function: no class, no side effects beyond datamodule construction.

    Args:
        settings: Inference job configuration (InferenceJobConfig or legacy
            InferenceJobConfig) with data sections.

    Returns:
        Configured LightningDataModule ready for predict_dataloader iteration.

    Raises:
        ValueError: If data sections are not configured.
    """
    try:
        from dlkit.infrastructure.io.locations import root as _root

        cfg_dir = _root()
    except Exception:
        cfg_dir = Path.cwd()

    context = BuildContext(mode="inference", working_directory=cfg_dir)

    if not isinstance(settings, JobConfig):
        raise ValueError(
            "build_inference_datamodule() requires an InferenceJobConfig or JobConfig instance. "
            "Legacy workflow config types are no longer supported."
        )
    data = settings.data
    if data is None:
        raise ValueError(
            "data section is required for batch inference. "
            "Ensure settings.data is configured before calling "
            "build_inference_datamodule()."
        )
    selected_features = tuple(data.features or ())
    selected_targets = tuple(data.targets or ())

    dataset = FlexibleDataset(
        entries=(*selected_features, *selected_targets),
    )

    split_cfg = data.splits
    session_name = settings.experiment.name if settings.experiment else "dlkit-experiment"
    split_resolution = get_or_create_split(
        num_samples=len(dataset),
        test_ratio=split_cfg.test_ratio,
        val_ratio=split_cfg.val_ratio,
        session_name=session_name,
        explicit_filepath=split_cfg.filepath,
    )
    dm_context = context.with_overrides(
        dataset=dataset,
        split=split_resolution.index_split,
        dataloader=None,
    )
    # Build from data.module selector
    module_settings = with_runtime_module_defaults(data.module)
    return FactoryProvider.create_component(cast(ComponentSettings, module_settings), dm_context)
