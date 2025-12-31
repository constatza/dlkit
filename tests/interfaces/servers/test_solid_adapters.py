"""Tests for SOLID adapter implementations."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from dlkit.interfaces.servers.tracking_adapter import FileBasedServerTracker
from dlkit.interfaces.servers.process_adapter import PsutilProcessKiller
from dlkit.interfaces.servers.storage_adapter import MLflowStorageSetup
from dlkit.interfaces.servers.infrastructure_adapters import (
    TyperUserInteraction,
    StandardFileSystemOperations,
    MLflowContextFactory,
)


class TestFileBasedServerTracker:
    """Test file-based server tracking implementation."""

    @pytest.fixture
    def tracker(self) -> FileBasedServerTracker:
        """Create server tracker instance."""
        return FileBasedServerTracker()

    @patch("dlkit.interfaces.servers.tracking_adapter.load_tracking_data")
    @patch("dlkit.interfaces.servers.tracking_adapter.save_tracking_data")
    @patch("dlkit.interfaces.servers.tracking_adapter.add_server_to_tracking")
    def test_track_server_calls_domain_functions_correctly(
        self,
        mock_add_server: Mock,
        mock_save_data: Mock,
        mock_load_data: Mock,
        tracker: FileBasedServerTracker,
    ) -> None:
        """Test that track_server orchestrates domain functions correctly."""
        mock_load_data.return_value = {"existing:8080": [99999]}
        mock_add_server.return_value = {"existing:8080": [99999], "localhost:5000": [12345]}

        tracker.track_server("localhost", 5000, 12345)

        # Verify load_tracking_data was called (path comes from _path_resolver)
        assert mock_load_data.call_count == 1
        mock_add_server.assert_called_once_with(
            {"existing:8080": [99999]}, "localhost", 5000, 12345
        )
        # Verify save_tracking_data was called (path comes from _path_resolver)
        assert mock_save_data.call_count == 1
        assert mock_save_data.call_args[0][1] == {"existing:8080": [99999], "localhost:5000": [12345]}

    @patch("dlkit.interfaces.servers.tracking_adapter.load_tracking_data")
    def test_track_server_handles_exceptions_silently(
        self,
        mock_load_data: Mock,
        tracker: FileBasedServerTracker,
    ) -> None:
        """Test that track_server handles exceptions silently."""
        mock_load_data.side_effect = Exception("File system error")

        # Should not raise exception
        tracker.track_server("localhost", 5000, 12345)

    @patch("dlkit.interfaces.servers.tracking_adapter.load_tracking_data")
    @patch("dlkit.interfaces.servers.tracking_adapter.save_tracking_data")
    @patch("dlkit.interfaces.servers.tracking_adapter.remove_server_from_tracking")
    def test_untrack_server_calls_domain_functions_correctly(
        self,
        mock_remove_server: Mock,
        mock_save_data: Mock,
        mock_load_data: Mock,
        tracker: FileBasedServerTracker,
        tmp_path: Path,
    ) -> None:
        """Test that untrack_server orchestrates domain functions correctly."""
        # Create a real tracking file so exists() check passes
        tracking_file = tmp_path / "servers.json"
        tracking_file.touch()

        # Mock path resolver to return our test path
        with patch.object(tracker._path_resolver, 'get_tracking_file_path', return_value=tracking_file):
            mock_load_data.return_value = {"localhost:5000": [12345]}
            mock_remove_server.return_value = {}

            tracker.untrack_server("localhost", 5000)

            mock_load_data.assert_called_once_with(tracking_file)
            mock_remove_server.assert_called_once_with({"localhost:5000": [12345]}, "localhost", 5000)
            mock_save_data.assert_called_once_with(tracking_file, {})

    def test_untrack_server_skips_if_file_missing(
        self,
        tracker: FileBasedServerTracker,
        tmp_path: Path,
    ) -> None:
        """Test that untrack_server skips when tracking file doesn't exist."""
        # Mock path resolver to return non-existent path
        nonexistent_path = tmp_path / "nonexistent.json"
        with patch.object(tracker._path_resolver, 'get_tracking_file_path', return_value=nonexistent_path):
            # Should not raise exception and should skip operations
            tracker.untrack_server("localhost", 5000)

    @patch("dlkit.interfaces.servers.tracking_adapter.load_tracking_data")
    @patch("dlkit.interfaces.servers.tracking_adapter.get_pids_for_server")
    def test_get_tracked_pids_calls_domain_functions_correctly(
        self,
        mock_get_pids: Mock,
        mock_load_data: Mock,
        tracker: FileBasedServerTracker,
        tmp_path: Path,
    ) -> None:
        """Test that get_tracked_pids orchestrates domain functions correctly."""
        # Create a real tracking file so exists() check passes
        tracking_file = tmp_path / "servers.json"
        tracking_file.touch()

        # Mock path resolver to return our test path
        with patch.object(tracker._path_resolver, 'get_tracking_file_path', return_value=tracking_file):
            mock_load_data.return_value = {"localhost:5000": [12345], "127.0.0.1:5000": [67890]}
            mock_get_pids.return_value = [12345, 67890]

            result = tracker.get_tracked_pids("localhost", 5000)

            assert result == [12345, 67890]
            mock_load_data.assert_called_once_with(tracking_file)
            mock_get_pids.assert_called_once_with(
                {"localhost:5000": [12345], "127.0.0.1:5000": [67890]}, "localhost", 5000
            )

    def test_get_tracked_pids_returns_empty_for_missing_file(
        self, tracker: FileBasedServerTracker, tmp_path: Path
    ) -> None:
        """Test that get_tracked_pids returns empty list for missing file."""
        # Mock path resolver to return non-existent path
        nonexistent_path = tmp_path / "nonexistent.json"
        with patch.object(tracker._path_resolver, 'get_tracking_file_path', return_value=nonexistent_path):
            result = tracker.get_tracked_pids("localhost", 5000)
            assert result == []


class TestPsutilProcessKiller:
    """Test psutil-based process killing implementation."""

    @pytest.fixture
    def mock_tracker(self) -> Mock:
        """Create mock server tracker."""
        return Mock()

    @pytest.fixture
    def killer(self, mock_tracker: Mock) -> PsutilProcessKiller:
        """Create process killer instance."""
        return PsutilProcessKiller(mock_tracker)

    def test_process_killer_requires_tracker_dependency(self) -> None:
        """Test that process killer requires tracker dependency."""
        mock_tracker = Mock()
        killer = PsutilProcessKiller(mock_tracker)

        assert killer._server_tracker is mock_tracker

    @patch("dlkit.interfaces.servers.process_adapter.psutil")
    def test_stop_server_processes_uses_tracked_pids_first(
        self,
        mock_psutil: Mock,
        killer: PsutilProcessKiller,
    ) -> None:
        """Test that process stopping tries tracked PIDs first."""
        import psutil as real_psutil

        killer._server_tracker.get_tracked_pids.return_value = [12345]

        # Mock valid MLflow process
        mock_proc = Mock()
        mock_proc.cmdline.return_value = [
            "python",
            "-m",
            "uvicorn",
            "mlflow.server:app",
            "--host",
            "localhost",
            "--port",
            "5000",
        ]

        # Set up Process mock to return process first, then raise NoSuchProcess for verification
        call_count = 0

        def process_side_effect(pid):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # validation and terminate calls
                return mock_proc
            else:  # verification calls after termination
                raise real_psutil.NoSuchProcess(pid)

        mock_psutil.Process.side_effect = process_side_effect
        mock_psutil.NoSuchProcess = real_psutil.NoSuchProcess

        # Mock process termination
        mock_proc.terminate.return_value = None
        mock_psutil.wait_procs.return_value = ([mock_proc], [])

        success = killer.stop_server_processes("localhost", 5000, False)

        assert success is True
        killer._server_tracker.get_tracked_pids.assert_called_once_with("localhost", 5000)

    @patch("dlkit.interfaces.servers.process_adapter.psutil")
    def test_stop_server_processes_scans_when_no_tracked_pids(
        self,
        mock_psutil: Mock,
        killer: PsutilProcessKiller,
    ) -> None:
        """Test that process stopping scans all processes when no tracked PIDs."""
        import psutil

        killer._server_tracker.get_tracked_pids.return_value = []

        # Mock process iteration
        mock_proc_info = {
            "pid": 99999,
            "cmdline": [
                "python",
                "-m",
                "uvicorn",
                "mlflow.server:app",
                "--host",
                "localhost",
                "--port",
                "5000",
            ],
        }
        mock_psutil.process_iter.return_value = [Mock(info=mock_proc_info)]

        # Mock process termination
        mock_proc = Mock()
        mock_psutil.Process.return_value = mock_proc
        mock_proc.terminate.return_value = None
        mock_psutil.wait_procs.return_value = ([mock_proc], [])

        # Mock NoSuchProcess for final verification (process should be gone)
        mock_psutil.NoSuchProcess = psutil.NoSuchProcess
        mock_psutil.Process.side_effect = [mock_proc, psutil.NoSuchProcess(99999)]

        success = killer.stop_server_processes("localhost", 5000, False)

        assert success is True
        mock_psutil.process_iter.assert_called_once()

    @patch("dlkit.interfaces.servers.process_adapter.psutil")
    def test_validate_tracked_processes_filters_non_mlflow(
        self,
        mock_psutil: Mock,
        killer: PsutilProcessKiller,
    ) -> None:
        """Test that tracked process validation filters non-MLflow processes."""
        # Mock mixed processes: one MLflow, one not
        mock_mlflow_proc = Mock()
        mock_mlflow_proc.cmdline.return_value = [
            "python",
            "-m",
            "uvicorn",
            "mlflow.server:app",
        ]

        mock_other_proc = Mock()
        mock_other_proc.cmdline.return_value = ["python", "other_app.py"]

        mock_psutil.Process.side_effect = [mock_mlflow_proc, mock_other_proc]

        messages = []
        valid_pids = killer._validate_tracked_processes([12345, 67890])

        assert valid_pids == [12345]  # Only MLflow process

    @patch("dlkit.interfaces.servers.process_adapter.psutil")
    def test_scan_for_mlflow_processes_finds_matching_processes(
        self,
        mock_psutil: Mock,
        killer: PsutilProcessKiller,
    ) -> None:
        """Test scanning finds processes matching host:port."""
        # Mock process with matching MLflow command
        mock_proc_info = {
            "pid": 99999,
            "cmdline": [
                "python",
                "-m",
                "uvicorn",
                "mlflow.server:app",
                "--host",
                "localhost",
                "--port",
                "5000",
            ],
        }
        mock_psutil.process_iter.return_value = [Mock(info=mock_proc_info)]

        pids = killer._scan_for_mlflow_processes("localhost", 5000)

        assert pids == [99999]

    @patch("dlkit.interfaces.servers.process_adapter.psutil")
    def test_terminate_processes_handles_graceful_shutdown(
        self,
        mock_psutil: Mock,
        killer: PsutilProcessKiller,
    ) -> None:
        """Test process termination with graceful shutdown."""
        import psutil

        mock_proc = Mock()
        mock_psutil.Process.return_value = mock_proc

        # Mock successful graceful termination
        mock_psutil.wait_procs.return_value = ([mock_proc], [])

        # Mock NoSuchProcess for final verification (process should be gone)
        mock_psutil.NoSuchProcess = psutil.NoSuchProcess
        mock_psutil.Process.side_effect = [mock_proc, psutil.NoSuchProcess(12345)]

        messages = []
        success = killer._terminate_processes([12345], False)

        assert success is True
        mock_proc.terminate.assert_called_once()
        mock_psutil.wait_procs.assert_called_once()

    @patch("dlkit.interfaces.servers.process_adapter.psutil")
    def test_terminate_processes_force_kills_stubborn_processes(
        self,
        mock_psutil: Mock,
        killer: PsutilProcessKiller,
    ) -> None:
        """Test process termination with force kill for stubborn processes."""
        import psutil

        mock_proc = Mock()

        # Mock failed graceful termination, successful force kill
        mock_psutil.wait_procs.side_effect = [
            ([], [mock_proc]),  # Graceful failed
            ([mock_proc], []),  # Force kill succeeded
        ]

        # Mock NoSuchProcess for final verification (process should be gone)
        mock_psutil.NoSuchProcess = psutil.NoSuchProcess
        # First call for terminate, second call for verification (which should raise NoSuchProcess)
        mock_psutil.Process.side_effect = [mock_proc, psutil.NoSuchProcess(12345)]

        messages = []
        success = killer._terminate_processes([12345], False)

        assert success is True
        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()
        assert mock_psutil.wait_procs.call_count == 2


class TestMLflowStorageSetup:
    """Test MLflow storage setup implementation."""

    @pytest.fixture
    def mock_user_interaction(self) -> Mock:
        """Create mock user interaction."""
        return Mock()

    @pytest.fixture
    def mock_file_system(self) -> Mock:
        """Create mock file system operations."""
        return Mock()

    @pytest.fixture
    def storage_setup(
        self, mock_user_interaction: Mock, mock_file_system: Mock
    ) -> MLflowStorageSetup:
        """Create storage setup instance."""
        return MLflowStorageSetup(mock_user_interaction, mock_file_system)

    def test_storage_setup_requires_dependencies(self) -> None:
        """Test that storage setup requires dependency injection."""
        user_interaction = Mock()
        file_system = Mock()

        setup = MLflowStorageSetup(user_interaction, file_system)

        assert setup._user_interaction is user_interaction
        assert setup._file_system is file_system

    @patch("dlkit.interfaces.servers.storage_adapter.should_use_default_storage")
    def test_ensure_storage_setup_skips_when_not_needed(
        self,
        mock_should_use_default: Mock,
        storage_setup: MLflowStorageSetup,
    ) -> None:
        """Test storage setup skips when default storage not needed."""
        mock_should_use_default.return_value = False
        server_config = Mock()
        overrides = {"host": "localhost"}

        result = storage_setup.ensure_storage_setup(server_config, overrides)

        assert result is server_config  # Unchanged
        mock_should_use_default.assert_called_once_with(server_config, overrides)

    @patch("dlkit.interfaces.servers.storage_adapter.should_use_default_storage")
    def test_ensure_storage_setup_uses_existing_directory(
        self,
        mock_should_use_default: Mock,
        mock_file_system: Mock,
        storage_setup: MLflowStorageSetup,
        tmp_path,
    ) -> None:
        """Test storage setup uses existing directory."""
        mock_should_use_default.return_value = True
        mock_path = tmp_path / "mlruns"
        mock_path.mkdir()

        # Mock the path resolver
        with patch.object(storage_setup._path_resolver, 'get_default_mlruns_path', return_value=mock_path):
            mock_file_system.directory_exists.return_value = True

            server_config = Mock()
            result = storage_setup.ensure_storage_setup(server_config, {})

        assert result is server_config
        mock_file_system.directory_exists.assert_called_once_with(mock_path)

    @patch("dlkit.interfaces.servers.storage_adapter.should_use_default_storage")

    def test_ensure_storage_setup_creates_directory_automatically(
        self,

        mock_should_use_default: Mock,
        mock_user_interaction: Mock,
        mock_file_system: Mock,
        storage_setup: MLflowStorageSetup,
    ) -> None:
        """Test storage setup creates directory automatically (no user interaction)."""
        mock_should_use_default.return_value = True
        mock_path = Mock()
        mock_file_system.directory_exists.return_value = False

        # Mock the path resolver to return our mock path
        with patch.object(storage_setup._path_resolver, 'get_default_mlruns_path', return_value=mock_path):
            server_config = Mock()
            result = storage_setup.ensure_storage_setup(server_config, {})

        assert result is server_config
        # Storage adapter is now pure business logic - no user interaction
        mock_user_interaction.show_message.assert_not_called()
        mock_user_interaction.confirm_action.assert_not_called()
        mock_file_system.create_directory.assert_called_once_with(mock_path)

    @patch("dlkit.interfaces.servers.storage_adapter.should_use_default_storage")

    def test_ensure_storage_setup_skips_when_directory_exists(
        self,

        mock_should_use_default: Mock,
        mock_user_interaction: Mock,
        mock_file_system: Mock,
        storage_setup: MLflowStorageSetup,
    ) -> None:
        """Test storage setup skips creation when directory already exists."""
        mock_should_use_default.return_value = True
        mock_path = Mock()
        mock_file_system.directory_exists.return_value = True  # Directory already exists

        # Mock the path resolver to return our mock path
        with patch.object(storage_setup._path_resolver, 'get_default_mlruns_path', return_value=mock_path):
            server_config = Mock()
            result = storage_setup.ensure_storage_setup(server_config, {})

        assert result is server_config
        # Should not attempt to create directory when it already exists
        mock_file_system.create_directory.assert_not_called()


class TestTyperUserInteraction:
    """Test Typer-based user interaction implementation."""

    @pytest.fixture
    def user_interaction(self) -> TyperUserInteraction:
        """Create user interaction instance."""
        return TyperUserInteraction()

    def test_confirm_action_returns_default_in_test_environment(
        self,
        user_interaction: TyperUserInteraction,
    ) -> None:
        """Test that confirm_action returns default value in test environment."""
        # In test environment, should return default without prompting
        result = user_interaction.confirm_action("Confirm?", default=True)
        assert result is True

        result = user_interaction.confirm_action("Confirm?", default=False)
        assert result is False

    def test_confirm_action_with_auto_confirm_flag(
        self,
        user_interaction: TyperUserInteraction,
    ) -> None:
        """Test that confirm_action respects auto_confirm flag."""
        # With auto_confirm=True, should return default without prompting
        result = user_interaction.confirm_action("Confirm?", default=True, auto_confirm=True)
        assert result is True

        result = user_interaction.confirm_action("Confirm?", default=False, auto_confirm=True)
        assert result is False

    @patch("rich.console.Console")
    def test_show_message_uses_rich_console(
        self,
        mock_console_class: Mock,
        user_interaction: TyperUserInteraction,
    ) -> None:
        """Test that show_message uses Rich console."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        user_interaction.show_message("Test message")

        mock_console.print.assert_called_once_with("Test message")


class TestStandardFileSystemOperations:
    """Test standard file system operations implementation."""

    @pytest.fixture
    def file_system(self) -> StandardFileSystemOperations:
        """Create file system operations instance."""
        return StandardFileSystemOperations()

    def test_create_directory_creates_path(
        self, file_system: StandardFileSystemOperations, tmp_path: Path
    ) -> None:
        """Test directory creation."""
        test_dir = tmp_path / "test" / "nested"

        file_system.create_directory(test_dir)

        assert test_dir.exists()
        assert test_dir.is_dir()

    def test_directory_exists_checks_correctly(
        self, file_system: StandardFileSystemOperations, tmp_path: Path
    ) -> None:
        """Test directory existence checking."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()

        nonexistent_dir = tmp_path / "nonexistent"

        assert file_system.directory_exists(existing_dir) is True
        assert file_system.directory_exists(nonexistent_dir) is False


class TestMLflowContextFactory:
    """Test MLflow context factory implementation."""

    @pytest.fixture
    def context_factory(self) -> MLflowContextFactory:
        """Create context factory instance."""
        return MLflowContextFactory()

    @patch("dlkit.interfaces.servers.server_configuration.validate_mlflow_config")
    @patch("dlkit.interfaces.servers.mlflow_adapter.MLflowServerContext")
    def test_create_server_context_validates_and_creates(
        self,
        mock_context_class: Mock,
        mock_validate: Mock,
        context_factory: MLflowContextFactory,
    ) -> None:
        """Test server context creation validates config and creates context."""
        mock_settings = Mock()
        mock_settings.server = Mock()
        mock_context = Mock()
        mock_context_class.return_value = mock_context

        result = context_factory.create_server_context(mock_settings, host="localhost", port=8080)

        assert result is mock_context
        mock_validate.assert_called_once_with(mock_settings)
        mock_context_class.assert_called_once_with(
            mock_settings.server, host="localhost", port=8080
        )


class TestAdapterIntegration:
    """Test integration between different adapters."""

    def test_process_killer_integrates_with_tracker(self) -> None:
        """Test that process killer properly integrates with tracker."""
        mock_tracker = Mock()
        mock_tracker.get_tracked_pids.return_value = [12345]

        killer = PsutilProcessKiller(mock_tracker)

        # Verify dependency injection worked
        assert killer._server_tracker is mock_tracker

        # Verify interaction - mock psutil to find no processes
        with patch("dlkit.interfaces.servers.process_adapter.psutil") as mock_psutil:
            # Mock psutil exceptions
            mock_psutil.NoSuchProcess = type('NoSuchProcess', (Exception,), {})
            mock_psutil.AccessDenied = type('AccessDenied', (Exception,), {})

            # Mock process_iter to return no processes
            mock_psutil.process_iter.return_value = []

            success = killer.stop_server_processes("localhost", 5000)

            # Should return True (idempotent - no processes found)
            assert success is True
            # Should have called tracker
            mock_tracker.get_tracked_pids.assert_called_once_with("localhost", 5000)

    def test_storage_setup_integrates_with_infrastructure(self) -> None:
        """Test that storage setup properly integrates with infrastructure."""
        mock_user_interaction = Mock()
        mock_file_system = Mock()

        storage_setup = MLflowStorageSetup(mock_user_interaction, mock_file_system)

        # Verify dependency injection worked
        assert storage_setup._user_interaction is mock_user_interaction
        assert storage_setup._file_system is mock_file_system

    def test_adapters_follow_single_responsibility_principle(self) -> None:
        """Test that each adapter has single responsibility."""
        # Tracker only tracks
        tracker = FileBasedServerTracker()
        assert hasattr(tracker, "track_server")
        assert hasattr(tracker, "untrack_server")
        assert hasattr(tracker, "get_tracked_pids")

        # Process killer only kills processes
        killer = PsutilProcessKiller(Mock())
        assert hasattr(killer, "stop_server_processes")

        # Storage setup only handles storage
        storage = MLflowStorageSetup(Mock(), Mock())
        assert hasattr(storage, "ensure_storage_setup")

        # User interaction only handles UI
        ui = TyperUserInteraction()
        assert hasattr(ui, "confirm_action")
        assert hasattr(ui, "show_message")

        # File system only handles files
        fs = StandardFileSystemOperations()
        assert hasattr(fs, "create_directory")
        assert hasattr(fs, "directory_exists")
