"""Pure factory for building inference datamodules without training components."""

from __future__ import annotations

from pathlib import Path

from lightning.pytorch import LightningDataModule

from dlkit.infrastructure.config.job_config import InferenceJobConfig, JobConfig

from ._dataset_helpers import flexible_dataset_overrides
from .datamodule_resolution import build_datamodule_from_selector
from .dataset_builder import DatasetBuilder


def build_inference_datamodule(
    settings: InferenceJobConfig | object,
    *,
    checkpoint_override: Path | str | None = None,
) -> LightningDataModule:
    """Build a datamodule for inference batch iteration.

    No training wrapper, no loss, no optimizer. Only run/experiment, data sections.
    Pure function: no class, no side effects beyond datamodule construction.

    Args:
        settings: Inference job configuration (InferenceJobConfig or legacy
            InferenceJobConfig) with data sections.
        checkpoint_override: Checkpoint path supplied directly by the caller
            (e.g. ``evaluate(checkpoint_path=...)`` or the CLI's required
            ``CHECKPOINT`` argument), used to auto-locate a colocated split
            file when ``data.splits.filepath`` is unset. Takes precedence
            over ``settings.model.checkpoint``, mirroring the same override
            precedence ``load_model_from_settings`` already applies for
            checkpoint/weight loading.

    Returns:
        Configured LightningDataModule ready for predict_dataloader iteration.

    Raises:
        ValueError: If data sections are not configured.
    """
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

    dataset_builder = DatasetBuilder()
    context = dataset_builder.build_context(settings)
    overrides = flexible_dataset_overrides(tuple(data.features or ()), tuple(data.targets or ()))
    dataset = dataset_builder.build_dataset(settings, context, overrides)
    split_resolution = dataset_builder.build_split_for_evaluation(
        settings, dataset, checkpoint_override=checkpoint_override
    )
    return build_datamodule_from_selector(data, dataset, split_resolution, context)
