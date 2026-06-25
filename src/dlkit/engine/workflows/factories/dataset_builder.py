"""Dataset and datamodule construction helpers for workflow strategies."""

from __future__ import annotations

from collections.abc import Sized
from pathlib import Path
from typing import Any, cast

from lightning.pytorch import LightningDataModule

from dlkit.engine.artifacts import (
    ContentArtifactPayload,
    FileArtifactPayload,
    ProducedArtifact,
)
from dlkit.engine.workflows.selectors.family_selector import DatasetFamilySelector
from dlkit.infrastructure.config.core.context import BuildContext
from dlkit.infrastructure.config.core.factories import FactoryProvider
from dlkit.infrastructure.config.enums import DatasetFamily
from dlkit.infrastructure.config.job_config import JobConfig
from dlkit.infrastructure.io.split_provider import SplitResolution, get_or_create_split

from .module_defaults import with_runtime_module_defaults

_DATAMODULE_MODULE = "dlkit.engine.adapters.lightning.datamodules"


def _is_inference_mode(settings: JobConfig) -> bool:
    """Return True when the workflow is in inference mode.

    Args:
        settings: A JobConfig instance.

    Returns:
        True if the run type is ``"predict"``.
    """
    return settings.run.type == "predict"


def _get_session_name(settings: JobConfig) -> str | None:
    """Extract session/experiment name for split namespacing.

    Args:
        settings: A JobConfig instance.

    Returns:
        Experiment name, or None when not configured.
    """
    return settings.experiment.name if settings.experiment else None


class DatasetBuilder:
    """Build datasets, splits, and datamodules for runtime workflows."""

    def build_context(self, settings: JobConfig) -> BuildContext:
        """Create the shared build context for a workflow.

        Args:
            settings: A JobConfig instance.

        Returns:
            BuildContext with mode and working directory resolved.
        """
        mode = "inference" if _is_inference_mode(settings) else "training"
        try:
            from dlkit.infrastructure.io.locations import root as root_path

            working_directory = root_path()
        except Exception:
            working_directory = Path.cwd()
        return BuildContext(mode=mode, working_directory=working_directory)

    def build_dataset(
        self,
        settings: JobConfig,
        context: BuildContext,
        overrides: dict[str, Any],
    ) -> object:
        """Build any dataset by applying caller-supplied overrides. No family branching.

        Args:
            settings: Full job configuration.
            context: Build context with resolved paths.
            overrides: Dict produced by the calling strategy. If the key ``"entries"``
                is present the overrides are for a FlexibleDataset; otherwise they
                are keyword arguments to a PyG/custom dataset constructor.

        Returns:
            Constructed dataset object.

        Raises:
            ValueError: If dataset settings are not configured or ``data.name`` is missing.
        """
        from dlkit.infrastructure.config.data_settings import DataSettings
        from dlkit.infrastructure.utils.general import import_object

        data = settings.data
        if data is None:
            raise ValueError("DATASET settings are required but not configured")

        if isinstance(data, DataSettings):
            ds_with_defaults = with_runtime_module_defaults(data)
            if "entries" in overrides:
                from dlkit.engine.data.datasets.flexible import FlexibleDataset

                return FlexibleDataset(entries=overrides["entries"])
            name = ds_with_defaults.name or data.name
            module_path = ds_with_defaults.module_path or data.module_path
            if name is None:
                raise ValueError("data.name (class) is required for dataset construction")
            dataset_cls = import_object(name, fallback_module=module_path or "")
            return dataset_cls(**overrides)

        ds_settings = with_runtime_module_defaults(data)
        return FactoryProvider.create_component(ds_settings, context.with_overrides(**overrides))

    def build_split(self, settings: JobConfig, dataset: object) -> SplitResolution:
        """Get or create the dataset split.

        Args:
            settings: A JobConfig instance.
            dataset: The constructed dataset (must be Sized).

        Returns:
            SplitResolution with index split and optional source path.

        Raises:
            ValueError: If dataset or split configuration is missing.
        """
        data = settings.data
        if data is None:
            raise ValueError("DATASET settings are required but not configured")
        split_cfg = data.splits
        if split_cfg is None:
            raise ValueError("Split configuration is required but not found in data settings")
        return get_or_create_split(
            num_samples=len(cast(Sized, dataset)),
            test_ratio=split_cfg.test_ratio,
            val_ratio=split_cfg.val_ratio,
            session_name=_get_session_name(settings) or "",
            explicit_filepath=split_cfg.filepath,
        )

    def build_datamodule(
        self,
        settings: JobConfig,
        context: BuildContext,
        dataset: object,
        split_resolution: SplitResolution,
        *,
        family: DatasetFamily | None = None,
    ) -> LightningDataModule:
        """Build the configured datamodule with optional family defaults.

        Args:
            settings: A JobConfig instance.
            context: Shared build context.
            dataset: The constructed dataset object.
            split_resolution: Resolved train/val/test split.
            family: Optional DatasetFamily for datamodule class defaults.

        Returns:
            Configured LightningDataModule.

        Raises:
            ValueError: If datamodule settings are not configured.
        """
        from dlkit.infrastructure.config.data_settings import DataSettings
        from dlkit.infrastructure.config.dataloader_settings import DataloaderSettings
        from dlkit.infrastructure.utils.general import import_object

        data = settings.data
        if isinstance(data, DataSettings):
            # New-style: DataSettings unifies dataset + datamodule.
            # Build the DataModule directly; DataModuleSelector is not a ComponentSettings.
            dm_selector = data.module
            module_path = dm_selector.module_path or _DATAMODULE_MODULE
            dm_name = dm_selector.name or "ArrayDataModule"
            # "InMemoryModule" is a placeholder default — map to the real class.
            if dm_name == "InMemoryModule":
                dm_name = "ArrayDataModule"
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
            if family is not None and dm_name in ("ArrayDataModule", "InMemoryModule"):
                dm_cls = DatasetFamilySelector.default_datamodule_class_for_family(family)
            else:
                dm_cls = import_object(dm_name, fallback_module=module_path)
            return dm_cls(
                dataset=dataset,
                split=split_resolution.index_split,
                dataloader=dataloader,
            )

        raise ValueError(
            "DataModule configuration requires a DataSettings (data section). "
            "Ensure the job config has a [data] section."
        )

    def build_split_artifact(self, split_resolution: SplitResolution) -> ProducedArtifact:
        """Create typed split artifact metadata for runtime tracking.

        Args:
            split_resolution: Resolved split with optional source path.

        Returns:
            ProducedArtifact with file or content payload.
        """
        if split_resolution.source_path is not None:
            return ProducedArtifact(
                kind="split",
                artifact_path=f"splits/{split_resolution.artifact_filename}",
                payload=FileArtifactPayload(file_path=split_resolution.source_path),
            )

        payload = split_resolution.index_split.model_dump_json(
            exclude_none=True,
            indent=2,
        )
        return ProducedArtifact(
            kind="split",
            artifact_path=f"splits/{split_resolution.artifact_filename}",
            payload=ContentArtifactPayload(content=payload),
        )
