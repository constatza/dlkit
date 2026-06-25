"""Generative workflow build strategies."""

from __future__ import annotations

from pathlib import Path

from dlkit.engine.artifacts import (
    ContentArtifactPayload,
    FileArtifactPayload,
    ProducedArtifact,
)
from dlkit.engine.training.components import RuntimeComponents
from dlkit.infrastructure.types.split import IndexSplit

from .build_strategy import IBuildStrategy, WorkflowSettings


class GenerativeBuildStrategy(IBuildStrategy):
    """Base class for workflows with a GENERATIVE section."""

    def can_handle(self, settings: WorkflowSettings) -> bool:
        try:
            return getattr(settings, "GENERATIVE", None) is not None
        except Exception:
            return False


class FlowMatchingBuildStrategy(GenerativeBuildStrategy):
    """Build strategy for flow matching generative models."""

    def can_handle(self, settings: WorkflowSettings) -> bool:
        if not super().can_handle(settings):
            return False
        try:
            generative_settings = getattr(settings, "GENERATIVE", None)
            return getattr(generative_settings, "algorithm", None) == "flow_matching"
        except Exception:
            return False

    def _build_core(self, settings: WorkflowSettings) -> RuntimeComponents:
        raise NotImplementedError(
            "FlowMatchingBuildStrategy does not yet support JobConfig. "
            "Implement a JobConfig-compatible datamodule builder to enable this workflow."
        )


def _build_split_artifact(
    split: IndexSplit,
    artifact_filename: str,
    source_path: Path | None,
) -> ProducedArtifact:
    artifact_path = f"splits/{artifact_filename}"
    if source_path is not None:
        return ProducedArtifact(
            kind="split",
            artifact_path=artifact_path,
            payload=FileArtifactPayload(file_path=source_path),
        )
    return ProducedArtifact(
        kind="split",
        artifact_path=artifact_path,
        payload=ContentArtifactPayload(content=split.model_dump_json(exclude_none=True, indent=2)),
    )
