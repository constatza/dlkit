"""Structured MLflow dataset logging strategies and the chain that runs them.

Contract (``IDatasetLoggingStrategy``): a strategy returns ``Logged`` on success and
``Declined`` when it does not apply; it raises on genuine failure and never catches.
``StructuredDatasetLogger`` owns the single error boundary, so one broken strategy
cannot stop the strategies after it.

Digest policy: strategies holding the data in memory (numpy, pandas) let MLflow compute
its own content digest. Only ``PointerDatasetLoggingStrategy``, which must not read the
data, uses the metadata-based ``LineageRecord.digest``.
"""

from __future__ import annotations

import warnings
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import mlflow.data
import numpy as np
from mlflow.data.dataset_source_registry import resolve_dataset_source
from mlflow.data.meta_dataset import MetaDataset
from mlflow.exceptions import MlflowException
from mlflow.types import ColSpec, Schema, TensorSpec

from dlkit.infrastructure.config.job_config import JobConfig
from dlkit.infrastructure.utils.logging_config import get_logger
from dlkit.infrastructure.zarr import ILazyReader

from .dataset_payload import EntryNumpyPayload, EntryNumpyPayloadBuilder
from .dataset_sources import compute_sources_fingerprint, path_entries

type _WorkflowSettings = JobConfig

logger = get_logger(__name__)

_LOG_CONTEXT = "training"
_AMBIGUOUS_SOURCE_WARNING = "The specified dataset source can be interpreted in multiple ways:.*"


@dataclass(frozen=True, slots=True, kw_only=True)
class LineageRecord:
    """Identity of the dataset being logged, shared by every strategy."""

    name: str
    sources: tuple[str, ...]
    tags: Mapping[str, str]

    @property
    def source(self) -> str | None:
        """Primary source path (MLflow records a single source per dataset)."""
        return self.sources[0] if self.sources else None

    @property
    def digest(self) -> str | None:
        """Metadata-based digest of all sources, or None when there are none."""
        return compute_sources_fingerprint(self.sources) if self.sources else None

    @property
    def mlflow_tags(self) -> dict[str, str] | None:
        """Tags in the shape ``log_dataset`` expects (None when empty)."""
        return dict(self.tags) or None


@dataclass(frozen=True, slots=True)
class Logged:
    """The dataset was logged to MLflow."""


@dataclass(frozen=True, slots=True)
class Declined:
    """The strategy does not apply, or failed; ``reason`` says why."""

    reason: str


type LineageOutcome = Logged | Declined


@runtime_checkable
class IDatasetLoggingStrategy(Protocol):
    """Strategy interface for structured MLflow dataset logging."""

    def log(
        self,
        *,
        dataset: Any,
        run_context: Any,
        settings: _WorkflowSettings,
        record: LineageRecord,
    ) -> LineageOutcome:
        """Log the dataset, or return ``Declined`` when this strategy does not apply.

        Raises:
            Exception: On genuine failure; the chain catches and records it.
        """
        ...


@contextmanager
def _quiet_source_warning() -> Generator[None]:
    """Silence MLflow's ambiguous-dataset-source warning for plain file paths."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_AMBIGUOUS_SOURCE_WARNING, category=UserWarning)
        yield


def _mlflow_from_numpy(payload: EntryNumpyPayload, record: LineageRecord) -> Any:
    """Build an MLflow numpy dataset (MLflow computes its own content digest).

    ``from_numpy`` is injected into ``mlflow.data``'s namespace at runtime by its
    dataset-constructor registry, so static analysis cannot see it.
    """
    return mlflow.data.from_numpy(  # ty: ignore[unresolved-attribute]
        features=payload.features,
        targets=payload.targets,
        name=record.name,
        source=record.source,
    )


def _mlflow_from_pandas(dataframe: Any, record: LineageRecord) -> Any:
    """Build an MLflow pandas dataset (MLflow computes its own content digest).

    ``from_pandas`` is injected at runtime like ``from_numpy``.
    """
    return mlflow.data.from_pandas(  # ty: ignore[unresolved-attribute]
        dataframe,
        name=record.name,
        source=record.source,
    )


class EntryStructuredDatasetLoggingStrategy:
    """Log in-memory Feature/Target entries as an MLflow numpy dataset."""

    def __init__(self, payload_builder: EntryNumpyPayloadBuilder | None = None) -> None:
        self._payload_builder = payload_builder or EntryNumpyPayloadBuilder()

    def log(
        self,
        *,
        dataset: Any,
        run_context: Any,
        settings: _WorkflowSettings,
        record: LineageRecord,
    ) -> LineageOutcome:
        payload = self._payload_builder.build(settings)
        if payload is None:
            return Declined("no in-memory-only feature/target entries")

        with _quiet_source_warning():
            mlflow_dataset = _mlflow_from_numpy(payload, record)
            run_context.log_dataset(mlflow_dataset, context=_LOG_CONTEXT, tags=record.mlflow_tags)
        return Logged()


class TabularStructuredDatasetLoggingStrategy:
    """Log datasets exposing a tabular ``df`` attribute as an MLflow pandas dataset."""

    def log(
        self,
        *,
        dataset: Any,
        run_context: Any,
        settings: _WorkflowSettings,
        record: LineageRecord,
    ) -> LineageOutcome:
        dataframe = getattr(dataset, "df", None)
        if dataframe is None:
            return Declined("dataset has no `df` attribute")

        if hasattr(dataframe, "to_pandas"):
            dataframe = dataframe.to_pandas()

        with _quiet_source_warning():
            mlflow_dataset = _mlflow_from_pandas(dataframe, record)
            run_context.log_dataset(mlflow_dataset, context=_LOG_CONTEXT, tags=record.mlflow_tags)
        return Logged()


class LazyReaderSchemaBuilder:
    """Build an MLflow tensor schema from lazy readers without reading array data.

    ``ZarrLazyReader`` and ``Hdf5LazyReader`` probe shape/dtype once at construction,
    so the schema is free. It is all-or-nothing: a schema that covered only some of the
    file-backed entries would silently misdescribe the dataset, so any entry without a
    lazy reader (e.g. ``.npy``, ``.npz``, eager HDF5) yields no schema at all.
    """

    def build(self, settings: _WorkflowSettings) -> Schema | None:
        entries = path_entries(settings)
        if not entries:
            return None

        try:
            specs: list[ColSpec | TensorSpec] = []
            for entry in entries:
                reader = entry.open_reader()
                if not isinstance(reader, ILazyReader):
                    logger.debug(
                        "No lazy reader for entry '{}'; skipping lineage schema", entry.name
                    )
                    return None
                specs.append(
                    TensorSpec(
                        type=np.dtype(reader.dtype),
                        shape=(-1, *reader.sample_shape),
                        name=entry.name,
                    )
                )
        except (OSError, TypeError, ValueError, MlflowException) as exc:
            logger.warning("Could not build lineage schema from lazy readers: {}", exc)
            return None

        return Schema(specs)


class PointerDatasetLoggingStrategy:
    """Pointer-only lineage for sources that can't be eagerly materialized.

    Logs a real ``mlflow.data.Dataset`` (visible to MLflow's lineage tooling) built from
    just the source path and a metadata-based digest; no array data is read.
    """

    def __init__(self, schema_builder: LazyReaderSchemaBuilder | None = None) -> None:
        self._schema_builder = schema_builder or LazyReaderSchemaBuilder()

    def log(
        self,
        *,
        dataset: Any,
        run_context: Any,
        settings: _WorkflowSettings,
        record: LineageRecord,
    ) -> LineageOutcome:
        if record.source is None:
            return Declined("no source path to point at")

        schema = self._schema_builder.build(settings)

        with _quiet_source_warning():
            pointer_dataset = MetaDataset(  # ty: ignore[call-non-callable]
                resolve_dataset_source(record.source),
                name=record.name,
                digest=record.digest,
                schema=schema,
            )
            run_context.log_dataset(pointer_dataset, context=_LOG_CONTEXT, tags=record.mlflow_tags)
        return Logged()


class StructuredDatasetLogger:
    """Chain of responsibility over structured dataset logging strategies."""

    def __init__(self, strategies: list[IDatasetLoggingStrategy] | None = None) -> None:
        self._strategies: list[IDatasetLoggingStrategy] = strategies or [
            EntryStructuredDatasetLoggingStrategy(),
            TabularStructuredDatasetLoggingStrategy(),
            PointerDatasetLoggingStrategy(),
        ]

    def log(
        self,
        *,
        dataset: Any,
        run_context: Any,
        settings: _WorkflowSettings,
        record: LineageRecord,
    ) -> LineageOutcome:
        """Return ``Logged`` from the first succeeding strategy, else a combined ``Declined``."""
        reasons: list[str] = []
        for strategy in self._strategies:
            outcome = self._attempt(strategy, dataset, run_context, settings, record)
            match outcome:
                case Logged():
                    logger.debug("Logged dataset '{}' via {}", record.name, type(strategy).__name__)
                    return outcome
                case Declined(reason=reason):
                    reasons.append(f"{type(strategy).__name__}: {reason}")

        return Declined("; ".join(reasons) or "no strategies configured")

    @staticmethod
    def _attempt(
        strategy: IDatasetLoggingStrategy,
        dataset: Any,
        run_context: Any,
        settings: _WorkflowSettings,
        record: LineageRecord,
    ) -> LineageOutcome:
        try:
            return strategy.log(
                dataset=dataset, run_context=run_context, settings=settings, record=record
            )
        except Exception as exc:
            logger.warning(
                "Structured dataset logging strategy '{}' failed: {}", type(strategy).__name__, exc
            )
            return Declined(f"raised {type(exc).__name__}: {exc}")
