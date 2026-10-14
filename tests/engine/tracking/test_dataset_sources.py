"""Tests for lineage source discovery and the metadata-based fingerprint."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from dlkit.engine.tracking.dataset_sources import (
    FINGERPRINT_HEX_LENGTH,
    MLFLOW_MAX_DIGEST_LENGTH,
    compute_sources_fingerprint,
    iter_entries,
    path_entries,
)
from dlkit.infrastructure.config.data_roles import DataRole
from dlkit.infrastructure.config.data_settings import DataSettings
from dlkit.infrastructure.config.entry_types import NpyEntry, ValueEntry
from dlkit.infrastructure.config.job_config import JobConfig
from dlkit.infrastructure.config.run_settings import RunSettings


def _job(features: tuple = (), targets: tuple = ()) -> JobConfig:
    return JobConfig(
        run=RunSettings(type="train"),
        data=DataSettings.model_validate({"class": "CustomDataset"}).model_copy(
            update={"features": features, "targets": targets}
        ),
    )


@pytest.fixture
def npy_file(tmp_path: Path) -> Path:
    path = tmp_path / "x.npy"
    np.save(path, np.zeros((2, 2), dtype="float32"))
    return path


class TestFingerprint:
    def test_order_independent(self, tmp_path: Path) -> None:
        a, b = str(tmp_path / "a"), str(tmp_path / "b")
        assert compute_sources_fingerprint([a, b]) == compute_sources_fingerprint([b, a])

    def test_missing_and_remote_sources_do_not_raise(self) -> None:
        digest = compute_sources_fingerprint(["s3://bucket/key", "/no/such/file"])
        assert len(digest) == FINGERPRINT_HEX_LENGTH

    def test_digest_fits_mlflow_limit(self) -> None:
        assert len(compute_sources_fingerprint(["/a"])) <= MLFLOW_MAX_DIGEST_LENGTH

    def test_rewritten_file_changes_digest(self, npy_file: Path) -> None:
        before = compute_sources_fingerprint([str(npy_file)])
        np.save(npy_file, np.zeros((5, 5), dtype="float32"))
        os.utime(npy_file, ns=(1, 1))

        assert compute_sources_fingerprint([str(npy_file)]) != before

    def test_unchanged_file_keeps_digest(self, npy_file: Path) -> None:
        sources = [str(npy_file)]
        assert compute_sources_fingerprint(sources) == compute_sources_fingerprint(sources)


class TestEntryViews:
    def test_no_data_settings_gives_empty(self) -> None:
        job = JobConfig(run=RunSettings(type="train"))
        assert iter_entries(job) == ()
        assert path_entries(job) == ()

    def test_features_then_targets(self, npy_file: Path) -> None:
        feature = ValueEntry(name="x", value=np.ones((2, 2)), data_role=DataRole.FEATURE)
        target = NpyEntry(name="y", path=npy_file, data_role=DataRole.TARGET)

        assert iter_entries(_job((feature,), (target,))) == (feature, target)

    def test_path_entries_exclude_in_memory(self, npy_file: Path) -> None:
        feature = ValueEntry(name="x", value=np.ones((2, 2)), data_role=DataRole.FEATURE)
        target = NpyEntry(name="y", path=npy_file, data_role=DataRole.TARGET)

        assert path_entries(_job((feature,), (target,))) == (target,)
