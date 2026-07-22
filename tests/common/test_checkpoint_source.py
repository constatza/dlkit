"""Construction, equality, and immutability tests for CheckpointSource sum type."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from dlkit.common.checkpoint_source import LatestRunCheckpoint, RunCheckpoint


@pytest.fixture
def run_checkpoint() -> RunCheckpoint:
    """Fixture for a RunCheckpoint instance."""
    return RunCheckpoint(run_id="test-run-123")


@pytest.fixture
def latest_run_checkpoint_default() -> LatestRunCheckpoint:
    """Fixture for a LatestRunCheckpoint with default experiment_name."""
    return LatestRunCheckpoint()


@pytest.fixture
def latest_run_checkpoint_named() -> LatestRunCheckpoint:
    """Fixture for a LatestRunCheckpoint with explicit experiment_name."""
    return LatestRunCheckpoint(experiment_name="test-experiment")


def test_run_checkpoint_construction(run_checkpoint: RunCheckpoint) -> None:
    """RunCheckpoint constructs correctly with run_id."""
    assert run_checkpoint.run_id == "test-run-123"


def test_latest_run_checkpoint_default_construction(
    latest_run_checkpoint_default: LatestRunCheckpoint,
) -> None:
    """LatestRunCheckpoint constructs with None experiment_name by default."""
    assert latest_run_checkpoint_default.experiment_name is None


def test_latest_run_checkpoint_named_construction(
    latest_run_checkpoint_named: LatestRunCheckpoint,
) -> None:
    """LatestRunCheckpoint constructs correctly with explicit experiment_name."""
    assert latest_run_checkpoint_named.experiment_name == "test-experiment"


def test_run_checkpoint_is_frozen(run_checkpoint: RunCheckpoint) -> None:
    """RunCheckpoint is frozen and raises FrozenInstanceError on mutation."""
    with pytest.raises(FrozenInstanceError):
        run_checkpoint.run_id = "different-run"  # type: ignore


def test_latest_run_checkpoint_is_frozen(
    latest_run_checkpoint_named: LatestRunCheckpoint,
) -> None:
    """LatestRunCheckpoint is frozen and raises FrozenInstanceError on mutation."""
    with pytest.raises(FrozenInstanceError):
        latest_run_checkpoint_named.experiment_name = "different-experiment"  # type: ignore


def test_run_checkpoint_equality() -> None:
    """Two RunCheckpoint instances with the same run_id compare equal."""
    cp1 = RunCheckpoint(run_id="test-run-123")
    cp2 = RunCheckpoint(run_id="test-run-123")
    assert cp1 == cp2


def test_run_checkpoint_inequality() -> None:
    """Two RunCheckpoint instances with different run_ids compare not equal."""
    cp1 = RunCheckpoint(run_id="test-run-123")
    cp2 = RunCheckpoint(run_id="test-run-456")
    assert cp1 != cp2


def test_latest_run_checkpoint_equality() -> None:
    """Two LatestRunCheckpoint instances with the same experiment_name compare equal."""
    cp1 = LatestRunCheckpoint(experiment_name="test-experiment")
    cp2 = LatestRunCheckpoint(experiment_name="test-experiment")
    assert cp1 == cp2
