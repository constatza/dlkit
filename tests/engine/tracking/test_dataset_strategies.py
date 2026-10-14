"""Tests for structured dataset logging strategies, schema building and the chain."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pytest
import zarr

from dlkit.engine.tracking.dataset_sources import compute_sources_fingerprint
from dlkit.engine.tracking.dataset_strategies import (
    Declined,
    EntryStructuredDatasetLoggingStrategy,
    LazyReaderSchemaBuilder,
    LineageOutcome,
    LineageRecord,
    Logged,
    PointerDatasetLoggingStrategy,
    StructuredDatasetLogger,
    TabularStructuredDatasetLoggingStrategy,
)
from dlkit.infrastructure.config.data_roles import DataRole
from dlkit.infrastructure.config.data_settings import DataSettings
from dlkit.infrastructure.config.entry_types import Hdf5Entry, NpyEntry, ValueEntry, ZarrEntry
from dlkit.infrastructure.config.job_config import JobConfig
from dlkit.infrastructure.config.run_settings import RunSettings


class _RunContext:
    def __init__(self) -> None:
        self.logged: list[dict[str, Any]] = []

    def log_dataset(
        self, dataset: Any, context: str | None = None, tags: dict | None = None
    ) -> None:
        self.logged.append({"dataset": dataset, "context": context, "tags": tags})


def _job(features: tuple = (), targets: tuple = ()) -> JobConfig:
    return JobConfig(
        run=RunSettings(type="train"),
        data=DataSettings.model_validate({"class": "CustomDataset"}).model_copy(
            update={"features": features, "targets": targets}
        ),
    )


def _record(*sources: str, tags: dict[str, str] | None = None) -> LineageRecord:
    return LineageRecord(name="ds", sources=tuple(sources), tags=tags or {})


@pytest.fixture
def run_context() -> _RunContext:
    return _RunContext()


@pytest.fixture
def zarr_path(tmp_path: Path) -> Path:
    path = tmp_path / "array.zarr"
    zarr.open_array(str(path), mode="w", shape=(10, 5), chunks=(1, 5), dtype="float32")
    return path


@pytest.fixture
def npy_path(tmp_path: Path) -> Path:
    path = tmp_path / "x.npy"
    np.save(path, np.zeros((3, 2), dtype="float32"))
    return path


class TestLineageRecord:
    def test_source_and_digest_come_from_sources(self) -> None:
        record = _record("a", "b")

        assert record.source == "a"
        assert record.digest == compute_sources_fingerprint(["a", "b"])

    def test_empty_sources_have_no_source_or_digest(self) -> None:
        record = _record()

        assert record.source is None
        assert record.digest is None

    def test_empty_tags_become_none_for_mlflow(self) -> None:
        assert _record().mlflow_tags is None
        assert _record(tags={"k": "v"}).mlflow_tags == {"k": "v"}


class TestLazyReaderSchemaBuilder:
    def test_all_lazy_entries_give_full_schema(self, zarr_path: Path) -> None:
        entry = ZarrEntry(name="x", path=zarr_path, data_role=DataRole.FEATURE)

        schema = LazyReaderSchemaBuilder().build(_job((entry,)))

        assert schema is not None
        (spec,) = schema.inputs
        assert (spec.name, spec.shape, str(spec.type)) == ("x", (-1, 5), "float32")

    def test_lazy_hdf5_gives_schema(self, tmp_path: Path) -> None:
        path = tmp_path / "d.h5"
        with h5py.File(path, "w") as f:
            f.create_group("a").create_dataset("x", data=np.zeros((4, 3), dtype="float64"))
        entry = Hdf5Entry(
            name="x", path=path, group="a", key="x", data_role=DataRole.FEATURE, lazy=True
        )

        schema = LazyReaderSchemaBuilder().build(_job((entry,)))

        assert schema is not None
        assert str(schema.inputs[0].type) == "float64"

    def test_mixed_lazy_and_eager_gives_no_schema(self, zarr_path: Path, npy_path: Path) -> None:
        lazy = ZarrEntry(name="x", path=zarr_path, data_role=DataRole.FEATURE)
        eager = NpyEntry(name="y", path=npy_path, data_role=DataRole.TARGET, mmap=False)

        assert LazyReaderSchemaBuilder().build(_job((lazy,), (eager,))) is None

    def test_no_path_entries_gives_no_schema(self) -> None:
        assert LazyReaderSchemaBuilder().build(_job()) is None

    def test_unsupported_dtype_gives_no_schema_without_raising(
        self, zarr_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entry = ZarrEntry(name="x", path=zarr_path, data_role=DataRole.FEATURE)
        reader = entry.open_reader()
        monkeypatch.setattr(type(reader), "dtype", property(lambda self: "not-a-dtype"))

        assert LazyReaderSchemaBuilder().build(_job((entry,))) is None


class TestPointerStrategy:
    def test_declines_without_a_source(self, run_context: _RunContext) -> None:
        outcome = PointerDatasetLoggingStrategy().log(
            dataset=None, run_context=run_context, settings=_job(), record=_record()
        )

        assert isinstance(outcome, Declined)
        assert run_context.logged == []

    def test_logs_pointer_with_path_digest_and_tags(
        self, zarr_path: Path, run_context: _RunContext
    ) -> None:
        record = _record(str(zarr_path), tags={"k": "v"})

        outcome = PointerDatasetLoggingStrategy().log(
            dataset=None, run_context=run_context, settings=_job(), record=record
        )

        assert outcome == Logged()
        (logged,) = run_context.logged
        dataset = logged["dataset"]
        assert (dataset.name, dataset.digest, logged["tags"]) == ("ds", record.digest, {"k": "v"})
        assert str(zarr_path) in dataset.source.uri
        assert dataset.schema is None

    def test_attaches_schema_for_lazy_entries(
        self, zarr_path: Path, run_context: _RunContext
    ) -> None:
        entry = ZarrEntry(name="x", path=zarr_path, data_role=DataRole.FEATURE)

        PointerDatasetLoggingStrategy().log(
            dataset=None,
            run_context=run_context,
            settings=_job((entry,)),
            record=_record(str(zarr_path)),
        )

        assert run_context.logged[0]["dataset"].schema is not None


class TestInMemoryStrategiesUseMlflowDigest:
    """In-memory data keeps MLflow's content digest, so data changes stay detectable."""

    def _log_entry(self, value: np.ndarray, run_context: _RunContext) -> str:
        feature = ValueEntry(name="x", value=value, data_role=DataRole.FEATURE)
        record = _record("/same/path")
        outcome = EntryStructuredDatasetLoggingStrategy().log(
            dataset=None, run_context=run_context, settings=_job((feature,)), record=record
        )
        assert outcome == Logged()
        return run_context.logged[-1]["dataset"].digest

    def test_entry_digest_tracks_content_not_path(self, run_context: _RunContext) -> None:
        first = self._log_entry(np.zeros((4, 2)), run_context)
        second = self._log_entry(np.ones((4, 2)), run_context)

        assert first != second
        assert first != _record("/same/path").digest

    def test_entry_declines_for_path_based(self, npy_path: Path, run_context: _RunContext) -> None:
        entry = NpyEntry(name="x", path=npy_path, data_role=DataRole.FEATURE, mmap=False)

        outcome = EntryStructuredDatasetLoggingStrategy().log(
            dataset=None, run_context=run_context, settings=_job((entry,)), record=_record()
        )

        assert isinstance(outcome, Declined)

    def test_tabular_digest_tracks_content_not_path(self, run_context: _RunContext) -> None:
        def log(frame: pd.DataFrame) -> str:
            TabularStructuredDatasetLoggingStrategy().log(
                dataset=SimpleNamespace(df=frame),
                run_context=run_context,
                settings=_job(),
                record=_record("/same/path"),
            )
            return run_context.logged[-1]["dataset"].digest

        assert log(pd.DataFrame({"a": [1, 2]})) != log(pd.DataFrame({"a": [3, 4]}))

    def test_tabular_declines_without_df(self, run_context: _RunContext) -> None:
        outcome = TabularStructuredDatasetLoggingStrategy().log(
            dataset=SimpleNamespace(), run_context=run_context, settings=_job(), record=_record()
        )

        assert isinstance(outcome, Declined)


class _Scripted:
    """Strategy returning a fixed outcome (or raising), recording call order."""

    def __init__(self, result: LineageOutcome | Exception, calls: list[str], label: str) -> None:
        self._result, self._calls, self._label = result, calls, label

    def log(self, **kwargs: Any) -> LineageOutcome:
        self._calls.append(self._label)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class TestStructuredDatasetLoggerChain:
    def _run(self, *strategies: _Scripted) -> LineageOutcome:
        return StructuredDatasetLogger(strategies=list(strategies)).log(
            dataset=None, run_context=None, settings=_job(), record=_record()
        )

    def test_raising_strategy_falls_through(self) -> None:
        calls: list[str] = []

        outcome = self._run(
            _Scripted(RuntimeError("boom"), calls, "raise"),
            _Scripted(Logged(), calls, "ok"),
        )

        assert outcome == Logged()
        assert calls == ["raise", "ok"]

    def test_first_success_stops_the_chain(self) -> None:
        calls: list[str] = []

        self._run(_Scripted(Logged(), calls, "a"), _Scripted(Logged(), calls, "b"))

        assert calls == ["a"]

    def test_all_failing_returns_declined_with_every_reason(self) -> None:
        calls: list[str] = []

        outcome = self._run(
            _Scripted(RuntimeError("boom"), calls, "a"),
            _Scripted(Declined("not mine"), calls, "b"),
        )

        assert isinstance(outcome, Declined)
        assert "boom" in outcome.reason
        assert "not mine" in outcome.reason
        assert calls == ["a", "b"]
