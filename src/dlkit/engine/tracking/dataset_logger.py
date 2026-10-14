"""Dataset lineage logging service.

Single Responsibility: Orchestrate dataset lineage recording into an active run context.
Extracted from MLflowTracker so the tracker focuses on run lifecycle only.
"""

from __future__ import annotations

import json
from typing import Any

from dlkit.infrastructure.config.job_config import JobConfig
from dlkit.infrastructure.utils.logging_config import get_logger

from .dataset_sources import DatasetSourceCollector, compute_sources_fingerprint
from .dataset_strategies import Declined, LineageRecord, Logged, StructuredDatasetLogger
from .interfaces import IRunContext

type _WorkflowSettings = JobConfig

logger = get_logger(__name__)


class DatasetLogger:
    """Logs dataset lineage to an active run context.

    Attempts structured MLflow dataset logging first; falls back to a JSON
    manifest artifact so lineage is always captured.
    """

    def log_dataset_to_run(
        self, datamodule: Any, run_context: IRunContext, settings: _WorkflowSettings
    ) -> None:
        """Log dataset lineage for the given datamodule.

        Args:
            datamodule: Lightning datamodule (dataset retrieved via ``datamodule.dataset``).
            run_context: Active run context to log to.
            settings: Workflow settings for source path extraction and tagging.
        """
        dataset = getattr(datamodule, "dataset", None)
        tags = self._build_dataset_tags(settings, dataset)
        sources = self._collect_dataset_sources(settings, dataset)

        structured_logged = self._log_structured_dataset(
            dataset, run_context, settings, tags, sources
        )
        self._log_dataset_manifest_artifact(
            run_context=run_context,
            settings=settings,
            dataset=dataset,
            sources=sources,
            tags=tags,
            structured_logged=structured_logged,
        )

    def _log_structured_dataset(
        self,
        dataset: Any,
        run_context: IRunContext,
        settings: _WorkflowSettings,
        tags: dict[str, str],
        sources: list[str],
    ) -> bool:
        if dataset is None:
            logger.debug(
                "No dataset found in datamodule, continuing with settings-driven lineage logging"
            )

        record = LineageRecord(
            name=self._resolve_dataset_name(settings),
            sources=tuple(sources),
            tags=tags,
        )
        outcome = StructuredDatasetLogger().log(
            dataset=dataset, run_context=run_context, settings=settings, record=record
        )
        match outcome:
            case Logged():
                return True
            case Declined(reason=reason):
                logger.warning(
                    "Structured MLflow dataset logging unavailable for dataset class '{}' "
                    "({}); manifest artifact fallback will be used.",
                    type(dataset).__name__ if dataset is not None else "None",
                    reason,
                )
                return False

    def _resolve_dataset_name(self, settings: _WorkflowSettings) -> str:
        ds = settings.data
        configured_name = getattr(ds, "name", None) if ds is not None else None
        return str(configured_name) if configured_name else "training_data"

    def _build_dataset_tags(self, settings: _WorkflowSettings, dataset: Any) -> dict[str, str]:
        tags: dict[str, str] = {}
        if settings.data is not None:
            try:
                split_cfg = settings.data.splits
                if split_cfg is not None:
                    tags["split_test_ratio"] = str(split_cfg.test_ratio)
                    tags["split_val_ratio"] = str(split_cfg.val_ratio)
            except AttributeError as exc:
                logger.debug("Could not read split ratios for dataset tags: {}", exc)
            dataset_type = getattr(settings.data, "family", None)
            if dataset_type:
                tags["dataset_type"] = str(dataset_type)
        tags["dataset_class"] = type(dataset).__name__ if dataset is not None else "None"
        return tags

    def _collect_dataset_sources(self, settings: _WorkflowSettings, dataset: Any) -> list[str]:
        del dataset
        return DatasetSourceCollector().collect(settings)

    def _log_dataset_manifest_artifact(
        self,
        run_context: IRunContext,
        settings: _WorkflowSettings,
        dataset: Any,
        sources: list[str],
        tags: dict[str, str],
        structured_logged: bool,
    ) -> None:
        try:
            fingerprint = compute_sources_fingerprint(sources)

            manifest = {
                "dataset_name": self._resolve_dataset_name(settings),
                "dataset_class": type(dataset).__name__ if dataset is not None else None,
                "sources": sources,
                "source_count": len(sources),
                "fingerprint": fingerprint,
                "tags": tags,
                "structured_mlflow_dataset_logged": structured_logged,
            }

            run_context.log_artifact_content(
                json.dumps(manifest, indent=2, sort_keys=True),
                "lineage/dataset_manifest.json",
            )
            run_context.set_tag("dataset_manifest_artifact", "lineage")
            run_context.set_tag("dataset_source_count", str(len(sources)))
            run_context.set_tag("dataset_fingerprint", fingerprint)
        except Exception as e:
            logger.warning("Failed to log dataset manifest artifact: {}", e)
