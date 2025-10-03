"""Tests for pure domain functions."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from dlkit.interfaces.servers.domain_functions import (
    validate_mlflow_config,
    get_host_variants,
    get_tracking_file_path,
    load_tracking_data,
    save_tracking_data,
    should_use_default_storage,
    get_default_mlruns_path,
    is_mlflow_process,
    matches_host_port,
    add_server_to_tracking,
    remove_server_from_tracking,
    get_pids_for_server,
)


class TestMLflowConfigValidation:
    """Test MLflow configuration validation functions."""

    def test_validate_mlflow_config_with_valid_config(self) -> None:
        """Test validation passes with valid MLflow config."""
        config = Mock()
        config.enabled = True

        # Should not raise any exception
        validate_mlflow_config(config)

    def test_validate_mlflow_config_with_none_config(self) -> None:
        """Test validation fails with None config."""
        with pytest.raises(ValueError, match="MLFLOW configuration is missing"):
            validate_mlflow_config(None)

    def test_validate_mlflow_config_with_disabled_config(self) -> None:
        """Test validation fails with disabled MLflow."""
        config = Mock()
        config.enabled = False

        with pytest.raises(ValueError, match="MLflow is not enabled"):
            validate_mlflow_config(config)

    def test_validate_mlflow_config_with_missing_enabled_attribute(self) -> None:
        """Test validation fails when enabled attribute is missing."""
        config = Mock(spec=[])  # Mock with no attributes

        with pytest.raises(ValueError, match="MLflow is not enabled"):
            validate_mlflow_config(config)


class TestHostVariants:
    """Test host variant generation functions."""

    def test_get_host_variants_with_regular_host(self) -> None:
        """Test host variants generation with regular hostname."""
        variants = get_host_variants("myserver", 5000)

        assert "myserver:5000" in variants
        assert "127.0.0.1:5000" in variants
        assert "localhost:5000" in variants
        assert len(variants) == 3

    def test_get_host_variants_with_localhost(self) -> None:
        """Test host variants generation with localhost."""
        variants = get_host_variants("localhost", 8080)

        assert "localhost:8080" in variants
        assert "127.0.0.1:8080" in variants
        # Should not duplicate localhost
        assert variants.count("localhost:8080") == 1

    def test_get_host_variants_with_ip_address(self) -> None:
        """Test host variants generation with IP address."""
        variants = get_host_variants("127.0.0.1", 9000)

        assert "127.0.0.1:9000" in variants
        assert "localhost:9000" in variants
        # Should not duplicate 127.0.0.1
        assert variants.count("127.0.0.1:9000") == 1

    def test_get_host_variants_removes_duplicates(self) -> None:
        """Test that host variants removes duplicates while preserving order."""
        variants = get_host_variants("localhost", 5000)

        # Check no duplicates
        assert len(variants) == len(set(variants))

        # Check order preserved
        first_occurrence = {host: i for i, host in enumerate(variants)}
        assert all(i == variants.index(host) for host, i in first_occurrence.items())


class TestTrackingFileOperations:
    """Test tracking file path and I/O operations."""

    def test_get_tracking_file_path_returns_correct_path(self) -> None:
        """Test tracking file path generation."""
        path = get_tracking_file_path()

        assert path.name == "servers.json"
        assert path.parent.name == ".dlkit"
        assert path.parent.parent == Path.home()

    def test_load_tracking_data_with_missing_file(self, tmp_path: Path) -> None:
        """Test loading tracking dataflow when file doesn't exist."""
        nonexistent_file = tmp_path / "missing.json"

        result = load_tracking_data(nonexistent_file)

        assert result == {}

    def test_load_tracking_data_with_valid_file(self, tmp_path: Path) -> None:
        """Test loading tracking dataflow from valid JSON file."""
        tracking_file = tmp_path / "servers.json"
        test_data = {"localhost:5000": [12345, 67890], "127.0.0.1:8080": [11111]}

        with tracking_file.open("w") as f:
            json.dump(test_data, f)

        result = load_tracking_data(tracking_file)

        assert result == test_data

    def test_load_tracking_data_with_invalid_json(self, tmp_path: Path) -> None:
        """Test loading tracking dataflow from invalid JSON file."""
        tracking_file = tmp_path / "bad.json"
        tracking_file.write_text("{ invalid json")

        result = load_tracking_data(tracking_file)

        assert result == {}

    def test_load_tracking_data_sanitizes_values(self, tmp_path: Path) -> None:
        """Test that loading tracking dataflow sanitizes non-integer PIDs."""
        tracking_file = tmp_path / "servers.json"
        test_data = {
            "localhost:5000": [12345, "67890", 99999],
            "bad_server": "not_a_list",
            "127.0.0.1:8080": [11111, None, 22222],
        }

        with tracking_file.open("w") as f:
            json.dump(test_data, f)

        result = load_tracking_data(tracking_file)

        # Should clean up the dataflow
        expected = {
            "localhost:5000": [12345, 67890, 99999],  # string "67890" converted
            "127.0.0.1:8080": [11111, 22222],  # None filtered out
        }
        assert result == expected

    def test_save_tracking_data_creates_directory(self, tmp_path: Path) -> None:
        """Test that saving tracking dataflow creates parent directory."""
        nested_path = tmp_path / "nested" / "dir" / "servers.json"
        test_data = {"localhost:5000": [12345]}

        save_tracking_data(nested_path, test_data)

        assert nested_path.exists()
        assert nested_path.parent.exists()

    def test_save_tracking_data_writes_correct_format(self, tmp_path: Path) -> None:
        """Test that saving tracking dataflow writes correct JSON format."""
        tracking_file = tmp_path / "servers.json"
        test_data = {"localhost:5000": [12345, 67890], "127.0.0.1:8080": [11111]}

        save_tracking_data(tracking_file, test_data)

        # Read it back and verify
        with tracking_file.open("r") as f:
            saved_data = json.load(f)

        assert saved_data == test_data


class TestStorageDecisionLogic:
    """Test storage configuration decision logic."""

    def test_should_use_default_storage_with_explicit_backend_store(self) -> None:
        """Test storage decision when explicit backend store is provided."""
        server_config = Mock()
        server_config.backend_store_uri = None
        server_config.artifacts_destination = None

        overrides = {"backend_store_uri": "sqlite:///test.db"}

        result = should_use_default_storage(server_config, overrides)

        assert result is False

    def test_should_use_default_storage_with_explicit_artifacts(self) -> None:
        """Test storage decision when explicit artifacts destination is provided."""
        server_config = Mock()
        server_config.backend_store_uri = None
        server_config.artifacts_destination = None

        overrides = {"artifacts_destination": "/tmp/artifacts"}

        result = should_use_default_storage(server_config, overrides)

        assert result is False

    def test_should_use_default_storage_with_configured_backend_store(self) -> None:
        """Test storage decision when config has backend store configured."""
        server_config = Mock()
        server_config.backend_store_uri = "postgresql://host/db"
        server_config.artifacts_destination = None

        overrides = {}

        result = should_use_default_storage(server_config, overrides)

        assert result is False

    def test_should_use_default_storage_with_configured_artifacts(self) -> None:
        """Test storage decision when config has artifacts configured."""
        server_config = Mock()
        server_config.backend_store_uri = None
        server_config.artifacts_destination = "/config/artifacts"

        overrides = {}

        result = should_use_default_storage(server_config, overrides)

        assert result is False

    def test_should_use_default_storage_with_no_configuration(self) -> None:
        """Test storage decision when no storage is configured."""
        server_config = Mock()
        server_config.backend_store_uri = None
        server_config.artifacts_destination = None

        overrides = {}

        result = should_use_default_storage(server_config, overrides)

        assert result is True

    def test_get_default_mlruns_path_returns_current_directory(self) -> None:
        """Test that default MLruns path is in output subdirectory with environment awareness."""
        path = get_default_mlruns_path()

        assert path.name == "mlruns"
        # With environment awareness, path is resolved through resolver system
        # In tests, artifacts go under tests/artifacts; in production, under output/
        path_str = str(path.parent)
        assert "artifacts" in path_str or "output" in path_str
        assert path.is_absolute()


class TestProcessDetection:
    """Test process detection utility functions."""

    def test_is_mlflow_process_with_valid_mlflow_command(self) -> None:
        """Test MLflow process detection with valid command line."""
        cmdline = [
            "/usr/bin/python",
            "-m",
            "uvicorn",
            "mlflow.server:app",
            "--host",
            "0.0.0.0",
            "--port",
            "5000",
        ]

        result = is_mlflow_process(cmdline)

        assert result is True

    def test_is_mlflow_process_with_invalid_command(self) -> None:
        """Test MLflow process detection with non-MLflow command."""
        cmdline = ["/usr/bin/python", "my_app.py"]

        result = is_mlflow_process(cmdline)

        assert result is False

    def test_is_mlflow_process_with_empty_command(self) -> None:
        """Test MLflow process detection with empty command line."""
        result = is_mlflow_process([])

        assert result is False

    def test_is_mlflow_process_missing_uvicorn(self) -> None:
        """Test MLflow process detection without uvicorn."""
        cmdline = ["/usr/bin/python", "mlflow.server:app"]

        result = is_mlflow_process(cmdline)

        assert result is False

    def test_matches_host_port_with_matching_localhost(self) -> None:
        """Test host:port matching with localhost."""
        cmdline = [
            "/usr/bin/python",
            "-m",
            "uvicorn",
            "mlflow.server:app",
            "--host",
            "localhost",
            "--port",
            "5000",
        ]

        result = matches_host_port(cmdline, "localhost", 5000)

        assert result is True

    def test_matches_host_port_with_matching_ip(self) -> None:
        """Test host:port matching with IP address."""
        cmdline = [
            "/usr/bin/python",
            "-m",
            "uvicorn",
            "mlflow.server:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8080",
        ]

        result = matches_host_port(cmdline, "localhost", 8080)

        assert result is True  # Should match localhost variants

    def test_matches_host_port_with_non_matching_port(self) -> None:
        """Test host:port matching with different port."""
        cmdline = [
            "/usr/bin/python",
            "-m",
            "uvicorn",
            "mlflow.server:app",
            "--host",
            "localhost",
            "--port",
            "5000",
        ]

        result = matches_host_port(cmdline, "localhost", 8080)

        assert result is False

    def test_matches_host_port_with_empty_command(self) -> None:
        """Test host:port matching with empty command line."""
        result = matches_host_port([], "localhost", 5000)

        assert result is False


class TestTrackingDataManipulation:
    """Test tracking dataflow manipulation functions."""

    def test_add_server_to_tracking_new_server(self) -> None:
        """Test adding a new server to empty tracking"""
        servers = {}

        result = add_server_to_tracking(servers, "localhost", 5000, 12345)

        assert result == {"localhost:5000": [12345]}
        assert servers == {}  # Original should be unchanged

    def test_add_server_to_tracking_existing_server_new_pid(self) -> None:
        """Test adding a new PID to existing server."""
        servers = {"localhost:5000": [11111]}

        result = add_server_to_tracking(servers, "localhost", 5000, 12345)

        assert result == {"localhost:5000": [11111, 12345]}

    def test_add_server_to_tracking_existing_server_duplicate_pid(self) -> None:
        """Test adding duplicate PID to existing server."""
        servers = {"localhost:5000": [12345, 67890]}

        result = add_server_to_tracking(servers, "localhost", 5000, 12345)

        assert result == {"localhost:5000": [12345, 67890]}  # No duplicate

    def test_remove_server_from_tracking_existing_server(self) -> None:
        """Test removing existing server from tracking"""
        servers = {"localhost:5000": [12345], "127.0.0.1:5000": [67890], "otherhost:8080": [11111]}

        result = remove_server_from_tracking(servers, "localhost", 5000)

        # Should remove all variants of localhost:5000
        assert "localhost:5000" not in result
        assert "127.0.0.1:5000" not in result
        assert result == {"otherhost:8080": [11111]}

    def test_remove_server_from_tracking_nonexistent_server(self) -> None:
        """Test removing non-existent server from tracking"""
        servers = {"otherhost:8080": [11111]}

        result = remove_server_from_tracking(servers, "localhost", 5000)

        assert result == servers  # Unchanged

    def test_get_pids_for_server_with_multiple_variants(self) -> None:
        """Test getting PIDs for server with multiple host variants."""
        servers = {
            "localhost:5000": [12345, 23456],
            "127.0.0.1:5000": [34567],
            "otherhost:5000": [99999],
        }

        result = get_pids_for_server(servers, "localhost", 5000)

        # Should get PIDs from localhost variants
        assert set(result) == {12345, 23456, 34567}

    def test_get_pids_for_server_with_no_matches(self) -> None:
        """Test getting PIDs for server with no matches."""
        servers = {"otherhost:8080": [11111]}

        result = get_pids_for_server(servers, "localhost", 5000)

        assert result == []

    def test_get_pids_for_server_with_empty_tracking(self) -> None:
        """Test getting PIDs for server from empty tracking"""
        servers = {}

        result = get_pids_for_server(servers, "localhost", 5000)

        assert result == []


class TestFunctionPurity:
    """Test that domain functions are pure (no side effects)."""

    def test_functions_do_not_modify_input_parameters(self) -> None:
        """Test that functions don't modify their input parameters."""
        # Test tracking dataflow functions
        original_servers = {"localhost:5000": [12345]}
        servers_copy = original_servers.copy()

        add_server_to_tracking(servers_copy, "localhost", 8080, 67890)
        assert servers_copy == original_servers  # Unchanged

        remove_server_from_tracking(servers_copy, "localhost", 5000)
        assert servers_copy == original_servers  # Unchanged

        get_pids_for_server(servers_copy, "localhost", 5000)
        assert servers_copy == original_servers  # Unchanged

    def test_functions_are_deterministic(self) -> None:
        """Test that functions return same output for same input."""
        # Test host variants
        variants1 = get_host_variants("myhost", 5000)
        variants2 = get_host_variants("myhost", 5000)
        assert variants1 == variants2

        # Test storage decision
        config = Mock()
        config.backend_store_uri = None
        config.artifacts_destination = None
        overrides = {"host": "localhost"}

        result1 = should_use_default_storage(config, overrides)
        result2 = should_use_default_storage(config, overrides)
        assert result1 == result2

    def test_functions_handle_edge_cases_gracefully(self) -> None:
        """Test that functions handle edge cases without errors."""
        # Empty/None inputs
        assert get_host_variants("", 0) == [":0", "127.0.0.1:0", "localhost:0"]
        assert is_mlflow_process([]) is False
        assert matches_host_port([], "", 0) is False

        # Invalid server config
        assert should_use_default_storage(None, {}) is True  # Treats None as no config
