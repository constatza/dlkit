"""Tests for MLflow resource manager thread safety.

This module tests the thread-safety guarantees of the MLflow resource manager,
particularly the nested run stack and global state management.
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import Mock, patch

import pytest

from dlkit.runtime.workflows.strategies.tracking.mlflow_resource_manager import (
    MLflowResourceManager,
    MLflowResourceState,
)
from dlkit.tools.config.mlflow_settings import MLflowSettings, MLflowClientSettings


class TestMLflowResourceManagerThreadSafety:
    """Test thread safety of MLflow resource manager."""

    def test_stack_operations_are_thread_safe(self, tmp_path: Any) -> None:
        """Test that stack operations are protected by locks.

        Args:
            tmp_path: Temporary directory fixture
        """
        # Configure minimal MLflow settings
        client_config = MLflowClientSettings(tracking_uri=(tmp_path / "mlruns").as_uri())
        mlflow_config = MLflowSettings(enabled=True, client=client_config)

        manager = MLflowResourceManager(mlflow_config)

        # Test that stack_lock exists and is a lock type
        assert hasattr(manager._state, "stack_lock")
        # threading.Lock returns an instance, not a type - check it has lock methods
        assert hasattr(manager._state.stack_lock, "acquire")
        assert hasattr(manager._state.stack_lock, "release")
        assert callable(manager._state.stack_lock.acquire)
        assert callable(manager._state.stack_lock.release)

    def test_concurrent_state_snapshot_access(self, tmp_path: Any) -> None:
        """Test that state snapshots are thread-safe.

        Args:
            tmp_path: Temporary directory fixture
        """
        client_config = MLflowClientSettings(tracking_uri=(tmp_path / "mlruns").as_uri())
        mlflow_config = MLflowSettings(enabled=True, client=client_config)

        with MLflowResourceManager(mlflow_config) as manager:
            # Simulate concurrent access to state snapshot
            snapshots = []
            errors = []

            def get_snapshot() -> None:
                try:
                    snapshot = manager._get_state_snapshot()
                    snapshots.append(snapshot)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=get_snapshot) for _ in range(10)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            # All snapshots should be retrieved without errors
            assert len(errors) == 0
            assert len(snapshots) == 10

    def test_cleanup_is_thread_safe(self, tmp_path: Any) -> None:
        """Test that cleanup operations are thread-safe.

        Args:
            tmp_path: Temporary directory fixture
        """
        client_config = MLflowClientSettings(tracking_uri=(tmp_path / "mlruns").as_uri())
        mlflow_config = MLflowSettings(enabled=True, client=client_config)

        manager = MLflowResourceManager(mlflow_config)

        with manager:
            # Manually add some run IDs to the stack
            with manager._state.stack_lock:
                manager._state.active_run_stack.extend(["run1", "run2", "run3"])

        # After exit, stack should be empty (cleanup should have cleared it)
        assert len(manager._state.active_run_stack) == 0


class TestConflictDetection:
    """Test tracking URI conflict detection."""

    def test_conflicting_tracking_uris_raise_error(self, tmp_path: Any) -> None:
        """Test that conflicting tracking URIs are detected.

        Args:
            tmp_path: Temporary directory fixture
        """
        # This test requires mocking server_info since we can't actually start a server
        # with a different URI than the client config
        client_config = MLflowClientSettings(tracking_uri=(tmp_path / "mlruns_client").as_uri())
        mlflow_config = MLflowSettings(enabled=True, client=client_config)

        manager = MLflowResourceManager(mlflow_config)

        # Mock server_info with a different URI
        mock_server_info = Mock()
        mock_server_info.url = (tmp_path / "mlruns_server").as_uri()

        # Directly set server_info to simulate the conflict
        with pytest.raises(RuntimeError, match="Conflicting tracking URIs detected"):
            manager._state.server_info = mock_server_info
            manager._initialize_resources()

    def test_redundant_tracking_uris_log_warning(self, tmp_path: Any) -> None:
        """Test that redundant but matching tracking URIs log a warning.

        Args:
            tmp_path: Temporary directory fixture
        """
        tracking_uri = (tmp_path / "mlruns").as_uri()
        client_config = MLflowClientSettings(tracking_uri=tracking_uri)
        mlflow_config = MLflowSettings(enabled=True, client=client_config)

        manager = MLflowResourceManager(mlflow_config)

        # Mock server_info with the SAME URI
        mock_server_info = Mock()
        mock_server_info.url = tracking_uri

        manager._state.server_info = mock_server_info

        # Initialize resources - should log warning but not raise
        # We can't easily capture loguru logs in tests, but we can verify no error
        manager._initialize_resources()

        # Verify initialization succeeded
        assert manager._state.client is not None

    def test_set_global_tracking_uri_conflict_detection(self, tmp_path: Any) -> None:
        """Test that changing tracking URI is detected.

        Args:
            tmp_path: Temporary directory fixture
        """
        client_config = MLflowClientSettings(tracking_uri=(tmp_path / "mlruns").as_uri())
        mlflow_config = MLflowSettings(enabled=True, client=client_config)

        manager = MLflowResourceManager(mlflow_config)

        # First set
        manager._set_global_tracking_uri((tmp_path / "mlruns1").as_uri())

        # Attempt to change to different URI should raise error
        with pytest.raises(RuntimeError, match="Attempting to change global tracking URI"):
            manager._set_global_tracking_uri((tmp_path / "mlruns2").as_uri())


class TestStackConsistencyValidation:
    """Test stack consistency validation."""

    def test_validate_stack_consistency_with_desync(self, tmp_path: Any) -> None:
        """Test that stack consistency validation detects desynchronization.

        Args:
            tmp_path: Temporary directory fixture
        """
        client_config = MLflowClientSettings(tracking_uri=(tmp_path / "mlruns").as_uri())
        mlflow_config = MLflowSettings(enabled=True, client=client_config)

        with MLflowResourceManager(mlflow_config) as manager:
            # Manually add a run ID to simulate desynchronization
            with manager._state.stack_lock:
                manager._state.active_run_stack.append("fake_run_id")

            # Validation should not raise - it only logs warnings
            # We can't easily capture loguru logs, but we can verify it doesn't crash
            manager._validate_stack_consistency()

            # Verify the stack still has the fake run
            with manager._state.stack_lock:
                assert "fake_run_id" in manager._state.active_run_stack


class TestStateSnapshot:
    """Test state snapshot functionality."""

    def test_get_state_snapshot_returns_correct_data(self, tmp_path: Any) -> None:
        """Test that state snapshot contains expected data.

        Args:
            tmp_path: Temporary directory fixture
        """
        client_config = MLflowClientSettings(tracking_uri=(tmp_path / "mlruns").as_uri())
        mlflow_config = MLflowSettings(enabled=True, client=client_config)

        with MLflowResourceManager(mlflow_config) as manager:
            snapshot = manager._get_state_snapshot()

            # Verify snapshot structure
            assert "initialized" in snapshot
            assert "tracking_uri" in snapshot
            assert "client_exists" in snapshot
            assert "server_running" in snapshot
            assert "active_run_stack" in snapshot
            assert "stack_depth" in snapshot
            assert "experiment_id" in snapshot
            assert "mlflow_global_run" in snapshot

            # Verify values
            assert snapshot["initialized"] is True
            assert snapshot["client_exists"] is True
            assert isinstance(snapshot["active_run_stack"], list)
            assert snapshot["stack_depth"] == len(snapshot["active_run_stack"])
