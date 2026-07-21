"""Dataset and datamodule construction helpers for workflow strategies."""

from __future__ import annotations

from collections.abc import Sized
from pathlib import Path
from typing import Any, cast

from lightning.pytorch import LightningDataModule

from dlkit.common.errors import ConfigurationError
from dlkit.engine.artifacts import (
    ContentArtifactPayload,
    FileArtifactPayload,
    ProducedArtifact,
)
from dlkit.infrastructure.config.core.context import BuildContext
from dlkit.infrastructure.config.core.factories import FactoryProvider
from dlkit.infrastructure.config.enums import DatasetFamily
from dlkit.infrastructure.config.job_config import JobConfig
from dlkit.infrastructure.config.split_settings import IndexSplitSettings
from dlkit.infrastructure.io.split_provider import (
    SplitResolution,
    resolve_evaluation_split,
    resolve_training_split,
)

from .datamodule_resolution import build_datamodule_from_selector
from .module_defaults import with_runtime_module_defaults
from .run_output_paths import resolve_local_artifact_root

_DEFAULT_TEST_RATIO = IndexSplitSettings.model_fields["test_ratio"].default
_DEFAULT_VAL_RATIO = IndexSplitSettings.model_fields["val_ratio"].default


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

    def build_split_for_training(self, settings: JobConfig, dataset: object) -> SplitResolution:
        """Resolve (and locally persist) the split used by a training run.

        Producer path: used only by the train/optimize/converge build flow.
        Resolves the seed via ``settings.run.resolve_seed()`` and the local
        persistence path via ``resolve_local_artifact_root`` so evaluation
        can later reload the exact split used here instead of regenerating
        an unrelated random one.

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

        num_samples = len(cast(Sized, dataset))
        session_name = _get_session_name(settings) or "default"
        artifact_root = resolve_local_artifact_root(settings)
        persist_to = (
            artifact_root / "splits" / f"{session_name}_{num_samples}_split.json"
            if artifact_root is not None
            else None
        )

        return resolve_training_split(
            num_samples=num_samples,
            test_ratio=split_cfg.test_ratio,
            val_ratio=split_cfg.val_ratio,
            seed=settings.run.resolve_seed(),
            persist_to=persist_to,
            session_name=session_name,
            explicit_filepath=split_cfg.filepath,
            max_train_samples=split_cfg.max_train_samples,
            train_subset_seed=split_cfg.train_subset_seed,
        )

    def build_split_for_evaluation(
        self,
        settings: JobConfig,
        dataset: object,
        *,
        checkpoint_override: Path | str | None = None,
    ) -> SplitResolution:
        """Resolve the split used to reload a previously trained run's held-out set.

        Consumer path: used only by the inference/evaluation build flow.
        Never generates a fresh split — the structural ISP guarantee lives on
        ``resolve_evaluation_split`` itself, which has no ratio/seed
        parameters at all.

        Args:
            settings: A JobConfig instance (evaluation mode).
            dataset: The constructed dataset. Unused — kept for call-site
                symmetry with ``build_split_for_training``; evaluation never
                derives sample counts from the dataset since it only ever
                reloads an existing split file.
            checkpoint_override: Checkpoint path passed directly to
                ``evaluate()``/``predict`` (e.g. the CLI's required
                ``CHECKPOINT`` argument), used to auto-locate a colocated
                split file when ``data.splits.filepath`` is unset. Takes
                precedence over ``settings.model.checkpoint`` — this must
                mirror the same override precedence ``load_model_from_settings``
                already applies for checkpoint/weight loading, or the
                standard CLI invocation (checkpoint supplied only via the
                CLI argument, never duplicated into the config TOML) fails
                to resolve a split that verifiably exists on disk.

        Returns:
            SplitResolution loaded from the resolved split file.

        Raises:
            ConfigurationError: If ``data.splits.test_ratio``/``val_ratio``
                are set to non-default values (dead config for evaluation —
                the split is always reloaded verbatim), or if no split file
                can be resolved (neither an explicit ``data.splits.filepath``
                nor exactly one colocated ``splits/*.json`` sibling of the
                checkpoint).
        """
        del dataset
        data = settings.data
        explicit_filepath: Path | None = None
        if data is not None and data.splits is not None:
            split_cfg = data.splits
            ratios_overridden = (
                split_cfg.test_ratio != _DEFAULT_TEST_RATIO
                or split_cfg.val_ratio != _DEFAULT_VAL_RATIO
            )
            if ratios_overridden:
                raise ConfigurationError(
                    "data.splits.test_ratio/val_ratio are ignored during evaluation — "
                    "the split is always reloaded verbatim from a persisted file. "
                    "Remove these overrides from the evaluation config.",
                    {"test_ratio": split_cfg.test_ratio, "val_ratio": split_cfg.val_ratio},
                )
            explicit_filepath = split_cfg.filepath

        filepath = explicit_filepath or self._resolve_colocated_split_filepath(
            settings, checkpoint_override=checkpoint_override
        )
        return resolve_evaluation_split(filepath=filepath)

    def _resolve_colocated_split_filepath(
        self, settings: JobConfig, *, checkpoint_override: Path | str | None = None
    ) -> Path:
        """Auto-locate a unique ``splits/*.json`` file next to a checkpoint's run root.

        Args:
            settings: A JobConfig instance (evaluation mode).
            checkpoint_override: Checkpoint path passed directly by the
                caller (see ``build_split_for_evaluation``), taking
                precedence over ``settings.model.checkpoint``.

        Returns:
            Path to the single matching split file.

        Raises:
            ConfigurationError: If no checkpoint is configured, or zero/more
                than one candidate split file is found.
        """
        model = settings.model
        checkpoint = checkpoint_override or (model.checkpoint if model is not None else None)
        if checkpoint is None:
            raise ConfigurationError(
                "Cannot resolve an evaluation split: no data.splits.filepath was set "
                "and no model.checkpoint is configured to auto-locate a colocated "
                "splits/ directory. Set data.splits.filepath explicitly."
            )

        checkpoint_path = Path(checkpoint)
        if checkpoint_path.parent.name != "checkpoints":
            raise ConfigurationError(
                "Cannot auto-locate a colocated splits/ directory: checkpoint "
                f"{checkpoint_path!s} is not under a 'checkpoints/' directory "
                "(the convention resolve_local_artifact_root() writes to). "
                "Set data.splits.filepath explicitly.",
                {"checkpoint": str(checkpoint_path)},
            )

        run_root = checkpoint_path.parent.parent
        splits_dir = run_root / "splits"
        candidates = sorted(splits_dir.glob("*.json"))
        if len(candidates) == 1:
            return candidates[0]
        raise ConfigurationError(
            "Could not auto-resolve a unique evaluation split file next to "
            f"checkpoint {checkpoint!s}: found {len(candidates)} candidate(s) under "
            f"{splits_dir!s}. Set data.splits.filepath explicitly.",
            {"checkpoint": str(checkpoint), "candidates": [str(c) for c in candidates]},
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

        data = settings.data
        if isinstance(data, DataSettings):
            # New-style: DataSettings unifies dataset + datamodule.
            return build_datamodule_from_selector(
                data, dataset, split_resolution, context, family=family
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
