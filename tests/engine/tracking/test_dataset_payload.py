"""``EntryNumpyPayloadBuilder`` must never read a path-based entry's file content."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dlkit.engine.tracking.dataset_payload import EntryNumpyPayloadBuilder
from dlkit.infrastructure.config.data_roles import DataRole
from dlkit.infrastructure.config.data_settings import DataSettings
from dlkit.infrastructure.config.entry_types import NpyEntry, ValueEntry
from dlkit.infrastructure.config.job_config import JobConfig
from dlkit.infrastructure.config.run_settings import RunSettings


def _job(features: tuple, targets: tuple = ()) -> JobConfig:
    return JobConfig(
        run=RunSettings(type="train"),
        data=DataSettings.model_validate({"class": "CustomDataset"}).model_copy(
            update={"features": features, "targets": targets}
        ),
    )


@pytest.fixture
def builder() -> EntryNumpyPayloadBuilder:
    return EntryNumpyPayloadBuilder()


def test_all_in_memory_builds_payload(builder: EntryNumpyPayloadBuilder) -> None:
    feature = ValueEntry(
        name="x", value=np.ones((3, 2), dtype="float32"), data_role=DataRole.FEATURE
    )

    payload = builder.build(_job((feature,)))

    assert payload is not None
    np.testing.assert_array_equal(payload.features["x"], feature.value)
    assert payload.targets is None


def test_targets_are_materialized(builder: EntryNumpyPayloadBuilder) -> None:
    feature = ValueEntry(name="x", value=np.ones((3, 2)), data_role=DataRole.FEATURE)
    target = ValueEntry(name="y", value=np.zeros((3, 1)), data_role=DataRole.TARGET)

    payload = builder.build(_job((feature,), (target,)))

    assert payload is not None and payload.targets is not None
    assert set(payload.targets) == {"y"}


def test_any_path_based_entry_declines_entirely(
    tmp_path: Path, builder: EntryNumpyPayloadBuilder
) -> None:
    path = tmp_path / "y.npy"
    np.save(path, np.zeros((3, 1), dtype="float32"))
    feature = ValueEntry(name="x", value=np.ones((3, 2)), data_role=DataRole.FEATURE)
    target = NpyEntry(name="y", path=path, data_role=DataRole.TARGET, mmap=False)

    assert builder.build(_job((feature,), (target,))) is None


def test_no_data_settings_declines(builder: EntryNumpyPayloadBuilder) -> None:
    assert builder.build(JobConfig(run=RunSettings(type="train"))) is None
