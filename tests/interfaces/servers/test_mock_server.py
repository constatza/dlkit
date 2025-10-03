"""Fast mock server tests for MLflow functionality.

These tests use lightweight mocks to provide fast feedback on server
functionality without needing actual MLflow server processes.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from dlkit.interfaces.servers.factory import create_mlflow_adapter
from dlkit.interfaces.servers.mlflow_adapter import MLflowServerAdapter
from dlkit.interfaces.servers.protocols import ServerInfo, ServerStatus
from dlkit.tools.config.mlflow_settings import MLflowServerSettings


class MockMLflowServer:
    """Mock MLflow server for testing without real processes."""

    def __init__(self, host: str = "localhost", port: int = 5000) -> None:
        """Initialize mock server."""
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}"
        self.is_running = False
        self.start_count = 0
        self.stop_count = 0
        self.health_check_count = 0
        self._response_time = 0.1
        self._error_message: str | None = None

    def start(self) -> ServerInfo:
        """Start the mock server."""
        self.is_running = True
        self.start_count += 1
        return ServerInfo(
            process=Mock(pid=12345 + self.start_count),
            url=self.url,
            host=self.host,
            port=self.port,
            pid=12345 + self.start_count,
        )

    def stop(self) -> bool:
        """Stop the mock server."""
        self.is_running = False
        self.stop_count += 1
        return True

    def check_health(self) -> ServerStatus:
        """Check mock server health."""
        self.health_check_count += 1
        return ServerStatus(
            is_running=self.is_running,
            url=self.url,
            response_time=self._response_time if self.is_running else None,
            error_message=self._error_message if not self.is_running else None,
        )

    def set_unhealthy(self, error_message: str = "Connection refused") -> None:
        """Make the mock server appear unhealthy."""
        self.is_running = False
        self._error_message = error_message

    def set_slow_response(self, response_time: float) -> None:
        """Set mock server response time."""
        self._response_time = response_time


@pytest.fixture
def mock_server() -> MockMLflowServer:
    """Create mock MLflow server."""
    return MockMLflowServer()


@pytest.fixture
def mock_server_config(tmp_path: Path) -> MLflowServerSettings:
    """Create mock server configuration."""
    return MLflowServerSettings(
        scheme="http",
        host="localhost",
        port=5000,
        backend_store_uri=f"sqlite:///{(tmp_path / 'mock_mlflow.db').as_posix()}",
        artifacts_destination=str(tmp_path / "mock_artifacts"),
        num_workers=1,
    )


class TestMockServerBasics:
    """Test basic mock server functionality."""

    def test_mock_server_initialization(self, mock_server: MockMLflowServer) -> None:
        """Test mock server initialization."""
        assert mock_server.host == "localhost"
        assert mock_server.port == 5000
        assert mock_server.url == "http://localhost:5000"
        assert mock_server.is_running is False
        assert mock_server.start_count == 0
        assert mock_server.stop_count == 0
        assert mock_server.health_check_count == 0

    def test_mock_server_start_stop_cycle(self, mock_server: MockMLflowServer) -> None:
        """Test mock server start/stop cycle."""
        # Initially not running
        status = mock_server.check_health()
        assert status.is_running is False
        assert status.response_time is None

        # Start server
        server_info = mock_server.start()
        assert mock_server.is_running is True
        assert mock_server.start_count == 1
        assert server_info.url == "http://localhost:5000"
        assert server_info.pid == 12346  # 12345 + start_count

        # Check health while running
        status = mock_server.check_health()
        assert status.is_running is True
        assert status.response_time == 0.1
        assert status.error_message is None
        assert mock_server.health_check_count == 2

        # Stop server
        result = mock_server.stop()
        assert result is True
        assert mock_server.is_running is False
        assert mock_server.stop_count == 1

        # Check health after stop
        status = mock_server.check_health()
        assert status.is_running is False
        assert status.response_time is None

    def test_mock_server_multiple_starts(self, mock_server: MockMLflowServer) -> None:
        """Test mock server handles multiple starts."""
        # Start multiple times
        info1 = mock_server.start()
        info2 = mock_server.start()
        info3 = mock_server.start()

        assert mock_server.start_count == 3
        assert info1.pid == 12346
        assert info2.pid == 12347
        assert info3.pid == 12348

        # All should have same URL
        assert info1.url == info2.url == info3.url == "http://localhost:5000"

    def test_mock_server_health_scenarios(self, mock_server: MockMLflowServer) -> None:
        """Test different mock server health scenarios."""
        mock_server.start()

        # Healthy server
        status = mock_server.check_health()
        assert status.is_running is True
        assert status.response_time == 0.1

        # Set slow response
        mock_server.set_slow_response(2.5)
        status = mock_server.check_health()
        assert status.is_running is True
        assert status.response_time == 2.5

        # Set unhealthy
        mock_server.set_unhealthy("Server overloaded")
        status = mock_server.check_health()
        assert status.is_running is False
        assert status.error_message == "Server overloaded"
        assert status.response_time is None


class TestAdapterWithMockServer:
    """Test MLflow adapter integration with mock server."""

    def test_adapter_with_mock_server_lifecycle(
        self, mock_server: MockMLflowServer, mock_server_config: MLflowServerSettings
    ) -> None:
        """Test adapter lifecycle using mock server."""
        # Create adapter with mock dependencies
        mock_process_manager = Mock()
        mock_health_checker = Mock()

        adapter = MLflowServerAdapter(
            process_manager=mock_process_manager,
            health_checker=mock_health_checker,
            health_timeout=0.1,
        )

        # Configure mocks to simulate mock server behavior
        mock_health_checker.check_health.side_effect = lambda url: mock_server.check_health()
        mock_health_checker.wait_for_health.return_value = True
        mock_process_manager.start_process.side_effect = lambda config: mock_server.start().process
        mock_process_manager.stop_process.side_effect = lambda process: mock_server.stop()

        with patch.object(adapter, "_ensure_local_storage"):
            # Start server through adapter
            server_info = adapter.start_server(mock_server_config)

            assert server_info.url == "http://localhost:5000"
            assert mock_server.is_running is True
            assert mock_server.start_count == 1

            # Check server status through adapter
            status = adapter.check_server("localhost", 5000)
            assert status.is_running is True
            assert mock_server.health_check_count >= 1

            # Stop server through adapter
            result = adapter.stop_server(server_info)
            assert result is True
            assert mock_server.is_running is False
            assert mock_server.stop_count == 1

    def test_adapter_detects_already_running_mock_server(
        self, mock_server: MockMLflowServer, mock_server_config: MLflowServerSettings
    ) -> None:
        """Test adapter detects already running mock server."""
        # Pre-start mock server
        mock_server.start()

        mock_process_manager = Mock()
        mock_health_checker = Mock()
        mock_health_checker.check_health.side_effect = lambda url: mock_server.check_health()

        adapter = MLflowServerAdapter(
            process_manager=mock_process_manager, health_checker=mock_health_checker
        )

        # Adapter should detect existing server
        server_info = adapter.start_server(mock_server_config)

        assert server_info.url == "http://localhost:5000"
        assert server_info.process is None  # Not started by adapter
        assert mock_server.start_count == 1  # Only our pre-start

        # Should not have attempted to start new process
        mock_process_manager.start_process.assert_not_called()

    def test_adapter_handles_mock_server_startup_failure(
        self, mock_server: MockMLflowServer, mock_server_config: MLflowServerSettings
    ) -> None:
        """Test adapter handling mock server startup failure."""
        mock_process_manager = Mock()
        mock_health_checker = Mock()

        adapter = MLflowServerAdapter(
            process_manager=mock_process_manager,
            health_checker=mock_health_checker,
            health_timeout=0.1,
        )

        # Mock server not running initially
        mock_health_checker.check_health.side_effect = lambda url: mock_server.check_health()

        # Mock health check failure after "startup"
        def failing_wait_for_health(url: str, timeout: float) -> bool:
            mock_server.set_unhealthy("Failed to start properly")
            return False

        mock_health_checker.wait_for_health.side_effect = failing_wait_for_health
        mock_process_manager.start_process.side_effect = lambda config: mock_server.start().process
        mock_process_manager.stop_process.side_effect = lambda process: mock_server.stop()

        with patch.object(adapter, "_ensure_local_storage"):
            with pytest.raises(RuntimeError, match="MLflow server failed health check"):
                adapter.start_server(mock_server_config)

        # Should have attempted cleanup
        assert mock_server.stop_count >= 1


class TestConcurrentMockServerOperations:
    """Test concurrent operations with mock servers."""

    def test_multiple_adapters_with_shared_mock_server(
        self, mock_server_config: MLflowServerSettings
    ) -> None:
        """Test multiple adapters sharing a mock server."""
        shared_mock_server = MockMLflowServer()
        shared_mock_server.start()  # Pre-start server

        # Create multiple adapters
        adapters = []
        for i in range(3):
            mock_health_checker = Mock()
            mock_health_checker.check_health.side_effect = (
                lambda url: shared_mock_server.check_health()
            )

            adapter = MLflowServerAdapter(
                process_manager=Mock(), health_checker=mock_health_checker
            )
            adapters.append(adapter)

        # All adapters should detect the shared server
        server_infos = []
        for adapter in adapters:
            server_info = adapter.start_server(mock_server_config)
            server_infos.append(server_info)

        # All should reference the same server URL
        urls = [info.url for info in server_infos]
        assert all(url == "http://localhost:5000" for url in urls)

        # All should have process=None (not started by them)
        processes = [info.process for info in server_infos]
        assert all(proc is None for proc in processes)

        # Mock server should have been checked multiple times
        assert shared_mock_server.health_check_count >= 3

    def test_concurrent_health_checks_mock_server(self, mock_server: MockMLflowServer) -> None:
        """Test concurrent health checks on mock server."""
        mock_server.start()

        # Simulate concurrent health checks
        health_results = []

        def health_check_worker() -> None:
            """Worker function for health checking."""
            for _ in range(10):
                status = mock_server.check_health()
                health_results.append(status.is_running)
                time.sleep(0.0001)  # Small delay

        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=health_check_worker)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join(timeout=0.5)

        # All health checks should have succeeded
        assert len(health_results) == 50  # 5 threads * 10 checks each
        assert all(result is True for result in health_results)
        assert mock_server.health_check_count == 50

    def test_mock_server_state_transitions(self, mock_server: MockMLflowServer) -> None:
        """Test mock server state transitions under concurrent operations."""

        def start_stop_worker(iterations: int) -> None:
            """Worker that starts and stops server."""
            for _ in range(iterations):
                mock_server.start()
                time.sleep(0.0001)
                mock_server.stop()
                time.sleep(0.0001)

        def health_check_worker(iterations: int) -> None:
            """Worker that checks health."""
            for _ in range(iterations):
                mock_server.check_health()
                time.sleep(0.0001)

        # Start concurrent workers
        threads = [
            threading.Thread(target=start_stop_worker, args=(5,)),
            threading.Thread(target=health_check_worker, args=(20,)),
            threading.Thread(target=start_stop_worker, args=(3,)),
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join(timeout=0.5)

        # Verify final state and counters
        assert mock_server.start_count == 8  # 5 + 3 starts
        assert mock_server.stop_count == 8  # 5 + 3 stops
        assert mock_server.health_check_count == 20


class TestMockServerPerformance:
    """Test mock server performance characteristics."""

    def test_mock_server_response_time_simulation(self, mock_server: MockMLflowServer) -> None:
        """Test mock server response time simulation."""
        mock_server.start()

        # Test different response times
        response_times = [0.01, 0.1, 0.5, 1.0, 2.0]

        for expected_time in response_times:
            mock_server.set_slow_response(expected_time)
            status = mock_server.check_health()
            assert status.response_time == expected_time

    def test_mock_server_high_frequency_operations(self, mock_server: MockMLflowServer) -> None:
        """Test mock server with high frequency operations."""
        # Rapid start/stop cycles
        for _ in range(100):
            mock_server.start()
            mock_server.stop()

        assert mock_server.start_count == 100
        assert mock_server.stop_count == 100

        # Rapid health checks
        mock_server.start()
        for _ in range(1000):
            status = mock_server.check_health()
            assert status.is_running is True

        assert mock_server.health_check_count == 1000

    def test_mock_server_memory_usage_stability(self, mock_server: MockMLflowServer) -> None:
        """Test that mock server doesn't accumulate memory over many operations."""
        # This test ensures our mock doesn't have memory leaks
        initial_attrs = len(mock_server.__dict__)

        # Perform many operations
        for i in range(1000):
            mock_server.start()
            mock_server.check_health()
            mock_server.set_slow_response(0.1 + (i % 10) * 0.01)
            mock_server.set_unhealthy(f"Error {i}")
            mock_server.stop()

        # Object shouldn't grow in size
        final_attrs = len(mock_server.__dict__)
        assert final_attrs == initial_attrs


class TestMockServerIntegrationScenarios:
    """Test integration scenarios using mock servers."""

    def test_mock_training_workflow_with_mlflow_server(
        self, mock_server: MockMLflowServer, mock_server_config: MLflowServerSettings
    ) -> None:
        """Test mock training workflow that uses MLflow server."""
        # This simulates what a training workflow would do:
        # 1. Check if MLflow server is running
        # 2. Start server if needed
        # 3. Use server during training
        # 4. Clean up afterwards

        workflow_adapter = create_mlflow_adapter()

        # Mock the adapter's dependencies to use our mock server
        with (
            patch.object(workflow_adapter._health_checker, "check_health") as mock_check,
            patch.object(workflow_adapter._health_checker, "wait_for_health") as mock_wait,
            patch.object(workflow_adapter._process_manager, "start_process") as mock_start,
            patch.object(workflow_adapter._process_manager, "stop_process") as mock_stop,
            patch.object(workflow_adapter, "_ensure_local_storage"),
        ):
            mock_check.side_effect = lambda url: mock_server.check_health()
            mock_wait.return_value = True
            mock_start.side_effect = lambda config: mock_server.start().process
            mock_stop.side_effect = lambda process: mock_server.stop()

            # Workflow starts
            assert mock_server.is_running is False

            # Start MLflow server for training
            server_info = workflow_adapter.start_server(mock_server_config)
            assert mock_server.is_running is True
            assert mock_server.start_count == 1

            # Simulate training using the server (multiple health checks)
            for _ in range(5):
                status = workflow_adapter.check_server("localhost", 5000)
                assert status.is_running is True

            assert mock_server.health_check_count >= 5

            # Training completes, clean up server
            result = workflow_adapter.stop_server(server_info)
            assert result is True
            assert mock_server.is_running is False
            assert mock_server.stop_count == 1

    def test_mock_server_failure_recovery_workflow(
        self, mock_server: MockMLflowServer, mock_server_config: MLflowServerSettings
    ) -> None:
        """Test workflow handling mock server failures and recovery."""
        workflow_adapter = create_mlflow_adapter()

        with (
            patch.object(workflow_adapter._health_checker, "check_health") as mock_check,
            patch.object(workflow_adapter._health_checker, "wait_for_health") as mock_wait,
            patch.object(workflow_adapter._process_manager, "start_process") as mock_start,
            patch.object(workflow_adapter._process_manager, "stop_process") as mock_stop,
            patch.object(workflow_adapter, "_ensure_local_storage"),
        ):
            mock_check.side_effect = lambda url: mock_server.check_health()
            mock_start.side_effect = lambda config: mock_server.start().process
            mock_stop.side_effect = lambda process: mock_server.stop()

            # First attempt: server fails to start properly
            def failing_wait(url: str, timeout: float) -> bool:
                mock_server.set_unhealthy("Startup failed")
                return False

            mock_wait.side_effect = failing_wait

            with pytest.raises(RuntimeError, match="MLflow server failed health check"):
                workflow_adapter.start_server(mock_server_config)

            # Cleanup should have been attempted
            assert mock_server.stop_count >= 1

            # Second attempt: server starts successfully
            mock_server.is_running = False  # Reset state
            mock_wait.side_effect = lambda url, timeout: (mock_server.start(), True)[1]

            server_info = workflow_adapter.start_server(mock_server_config)
            assert server_info is not None
            assert mock_server.is_running is True

            # Clean up
            workflow_adapter.stop_server(server_info)
