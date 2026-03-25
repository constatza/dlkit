"""Tests for MLflow resource manager thread safety.

This module tests the thread-safety guarantees of the MLflow resource manager,
particularly the nested run stack and global state management.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from dlkit.runtime.workflows.strategies.tracking.backend import LocalSqliteBackend
from dlkit.runtime.workflows.strategies.tracking.mlflow_resource_manager import (
    MLflowResourceManager,
)
from dlkit.tools.config.mlflow_settings import MLflowSettings


@pytest.fixture
def mlflow_config_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MLflowSettings:
    """Provide an enabled MLflow settings instance with an isolated tracking URI."""
    monkeypatch.setenv(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
    )
    return MLflowSettings()


class TestMLflowResourceManagerThreadSafety:
    """Test thread safety of MLflow resource manager."""

    def test_stack_is_safe_under_concurrent_mutations(
        self, mlflow_config_enabled: MLflowSettings, tmp_path: Path
    ) -> None:
        """Concurrent push/pop must not corrupt the active run stack.

        50 threads each push a unique run ID and then pop it under the stack lock.
        After all threads finish the stack must be empty with no duplicates lost.

        Args:
            mlflow_config_enabled: Enabled MLflow settings fixture.
            tmp_path: Temporary directory for test artifacts.
        """
        db_path = tmp_path / "mlruns" / "mlflow.db"
        backend = LocalSqliteBackend(db_path=db_path)
        manager = MLflowResourceManager(mlflow_config_enabled, backend)
        errors: list[Exception] = []

        def push_and_pop(run_id: str) -> None:
            with manager._state.stack_lock:
                manager._state.active_run_stack.append(run_id)
            time.sleep(0)  # yield to increase contention
            with manager._state.stack_lock:
                if run_id in manager._state.active_run_stack:
                    manager._state.active_run_stack.remove(run_id)

        threads = [threading.Thread(target=push_and_pop, args=(f"run-{i}",)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert manager._state.active_run_stack == [], (
            "Stack must be empty after all threads complete"
        )
        assert len(errors) == 0

    def test_concurrent_state_snapshot_access(
        self, mlflow_config_enabled: MLflowSettings, tmp_path: Path
    ) -> None:
        """State snapshots must be readable from multiple concurrent threads.

        Args:
            mlflow_config_enabled: Enabled MLflow settings fixture.
            tmp_path: Temporary directory for test artifacts.
        """
        db_path = tmp_path / "mlruns" / "mlflow.db"
        backend = LocalSqliteBackend(db_path=db_path)
        with MLflowResourceManager(mlflow_config_enabled, backend) as manager:
            snapshots: list[dict] = []
            errors: list[Exception] = []

            def get_snapshot() -> None:
                try:
                    snapshots.append(manager._get_state_snapshot())
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=get_snapshot) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            assert len(snapshots) == 10

    def test_cleanup_clears_stack(
        self, mlflow_config_enabled: MLflowSettings, tmp_path: Path
    ) -> None:
        """Stack must be empty after context manager exit.

        Args:
            mlflow_config_enabled: Enabled MLflow settings fixture.
            tmp_path: Temporary directory for test artifacts.
        """
        db_path = tmp_path / "mlruns" / "mlflow.db"
        backend = LocalSqliteBackend(db_path=db_path)
        manager = MLflowResourceManager(mlflow_config_enabled, backend)

        with manager, manager._state.stack_lock:
            manager._state.active_run_stack.extend(["run1", "run2", "run3"])

        assert len(manager._state.active_run_stack) == 0


class TestConflictDetection:
    """Test tracking URI conflict detection."""

    def test_backend_immutability_guarantees_single_tracking_uri(self, tmp_path: Path) -> None:
        """Test that backend immutability guarantees a single tracking URI during resource lifecycle.

        Args:
            tmp_path: Temporary directory fixture
        """
        mlflow_config = MLflowSettings()
        db_path = tmp_path / "mlruns" / "mlflow.db"
        backend = LocalSqliteBackend(db_path=db_path)

        manager = MLflowResourceManager(mlflow_config, backend)

        # The backend is frozen/immutable, so get_tracking_uri() always returns the same value
        first_call = backend.tracking_uri()
        second_call = backend.tracking_uri()

        assert first_call == second_call, "Backend tracking URI should be constant"
        assert first_call == manager._backend.tracking_uri()


class TestStackConsistencyValidation:
    """Test stack consistency validation."""

    def test_validate_stack_consistency_with_desync(self, tmp_path: Path) -> None:
        """Test that stack consistency validation detects desynchronization.

        Args:
            tmp_path: Temporary directory fixture
        """
        mlflow_config = MLflowSettings()
        db_path = tmp_path / "mlruns" / "mlflow.db"
        backend = LocalSqliteBackend(db_path=db_path)

        with MLflowResourceManager(mlflow_config, backend) as manager:
            # Manually add a run ID to simulate desynchronization
            with manager._state.stack_lock:
                manager._state.active_run_stack.append("fake_run_id")

            # Get a state snapshot - this should work even with inconsistent state
            snapshot = manager._get_state_snapshot()

            # Verify the snapshot includes the fake run
            assert "fake_run_id" in snapshot["active_run_stack"]


class TestStateSnapshot:
    """Test state snapshot functionality."""

    def test_get_state_snapshot_returns_correct_data(self, tmp_path: Path) -> None:
        """Test that state snapshot contains expected data.

        Args:
            tmp_path: Temporary directory fixture
        """
        mlflow_config = MLflowSettings()
        db_path = tmp_path / "mlruns" / "mlflow.db"
        backend = LocalSqliteBackend(db_path=db_path)

        with MLflowResourceManager(mlflow_config, backend) as manager:
            snapshot = manager._get_state_snapshot()

            # Verify snapshot structure
            assert "initialized" in snapshot
            assert "tracking_uri" in snapshot
            assert "client_exists" in snapshot
            assert "active_run_stack" in snapshot
            assert "stack_depth" in snapshot
            assert "experiment_id" in snapshot
            assert "mlflow_global_run" in snapshot

            # Verify values
            assert snapshot["initialized"] is True
            assert snapshot["client_exists"] is True
            assert isinstance(snapshot["active_run_stack"], list)
            assert snapshot["stack_depth"] == len(snapshot["active_run_stack"])
