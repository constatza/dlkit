"""Typed artifact contracts for runtime production and publication.

Shared value objects used across engine.training, engine.tracking, and engine.adapters.
Lives at the engine top-level to avoid circular imports between those sub-packages.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

# Extensible scalar parameter value for runtime metadata surfaces.
# This is a sum type, not a renaming alias.
type ParamValue = str | int | float | bool

ArtifactKind = Literal[
    "split",
    "checkpoint",
    "prediction",
    "config",
    "dataset_manifest",
    "user_artifact",
]
TrackingBackendKind = Literal["none", "mlflow"]
PredictionPersistence = Literal["disabled", "local_only", "tracked"]
CheckpointPersistence = Literal["framework_local", "tracked"]
ConfigPersistence = Literal["none", "tracked", "local_explicit"]


@dataclass(frozen=True, slots=True, kw_only=True)
class FileArtifactPayload:
    file_path: Path


@dataclass(frozen=True, slots=True, kw_only=True)
class ContentArtifactPayload:
    content: str | bytes


type ArtifactPayload = FileArtifactPayload | ContentArtifactPayload


@dataclass(frozen=True, slots=True, kw_only=True)
class ProducedArtifact:
    kind: ArtifactKind
    artifact_path: str
    payload: ArtifactPayload


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactPolicy:
    tracking_backend: TrackingBackendKind = "none"
    prediction_persistence: PredictionPersistence = "disabled"
    checkpoint_persistence: CheckpointPersistence = "framework_local"
    config_persistence: ConfigPersistence = "none"
    local_root_dir: Path | None = None
    remove_uploaded_files: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeArtifactManifest:
    split_artifact: ProducedArtifact | None = None
    checkpoint_artifacts: tuple[ProducedArtifact, ...] = ()
    prediction_artifacts: tuple[ProducedArtifact, ...] = ()
    config_artifacts: tuple[ProducedArtifact, ...] = ()
    dataset_artifacts: tuple[ProducedArtifact, ...] = ()
    user_artifacts: tuple[ProducedArtifact, ...] = ()
    policy: ArtifactPolicy = field(default_factory=ArtifactPolicy)

    def with_policy(self, policy: ArtifactPolicy) -> RuntimeArtifactManifest:
        return RuntimeArtifactManifest(
            split_artifact=self.split_artifact,
            checkpoint_artifacts=self.checkpoint_artifacts,
            prediction_artifacts=self.prediction_artifacts,
            config_artifacts=self.config_artifacts,
            dataset_artifacts=self.dataset_artifacts,
            user_artifacts=self.user_artifacts,
            policy=policy,
        )

    def with_split_artifact(self, artifact: ProducedArtifact | None) -> RuntimeArtifactManifest:
        return RuntimeArtifactManifest(
            split_artifact=artifact,
            checkpoint_artifacts=self.checkpoint_artifacts,
            prediction_artifacts=self.prediction_artifacts,
            config_artifacts=self.config_artifacts,
            dataset_artifacts=self.dataset_artifacts,
            user_artifacts=self.user_artifacts,
            policy=self.policy,
        )

    def with_prediction_artifacts(
        self, artifacts: tuple[ProducedArtifact, ...]
    ) -> RuntimeArtifactManifest:
        return RuntimeArtifactManifest(
            split_artifact=self.split_artifact,
            checkpoint_artifacts=self.checkpoint_artifacts,
            prediction_artifacts=artifacts,
            config_artifacts=self.config_artifacts,
            dataset_artifacts=self.dataset_artifacts,
            user_artifacts=self.user_artifacts,
            policy=self.policy,
        )

    def with_checkpoint_artifacts(
        self, artifacts: tuple[ProducedArtifact, ...]
    ) -> RuntimeArtifactManifest:
        return RuntimeArtifactManifest(
            split_artifact=self.split_artifact,
            checkpoint_artifacts=artifacts,
            prediction_artifacts=self.prediction_artifacts,
            config_artifacts=self.config_artifacts,
            dataset_artifacts=self.dataset_artifacts,
            user_artifacts=self.user_artifacts,
            policy=self.policy,
        )


@runtime_checkable
class ArtifactPublisher(Protocol):
    def publish(self, artifact: ProducedArtifact) -> None: ...


@runtime_checkable
class ArtifactCollector(Protocol):
    def record(self, artifact: ProducedArtifact) -> None: ...

    def snapshot(self) -> tuple[ProducedArtifact, ...]: ...


@runtime_checkable
class NestedRunCapability(Protocol):
    def has_active_parent_run(self) -> bool: ...


@runtime_checkable
class ArtifactPolicyProvider(Protocol):
    def artifact_policy(self) -> ArtifactPolicy: ...


@runtime_checkable
class IMetricSink(Protocol):
    """Narrow protocol for consumers that only need to emit metrics, params, and tags.

    Components such as ``MLflowEpochLogger`` and ``MetricLogger`` depend on this
    interface rather than the full ``IRunContext``, satisfying ISP. Both ``IRunContext``
    implementations and ``NullRunContext`` satisfy this structurally.
    """

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...

    def log_params(self, params: Mapping[str, ParamValue]) -> None: ...

    def set_tag(self, key: str, value: str) -> None: ...


@dataclass(slots=True)
class InMemoryArtifactCollector:
    _artifacts: list[ProducedArtifact] = field(default_factory=list)

    def record(self, artifact: ProducedArtifact) -> None:
        self._artifacts.append(artifact)

    def snapshot(self) -> tuple[ProducedArtifact, ...]:
        return tuple(self._artifacts)
