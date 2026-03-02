"""Tests for mlflow_settings module.

This module tests the flattened MLflow settings classes for experiment tracking
and model registration functionality.
"""

from __future__ import annotations

import os
from typing import Any
import pytest
from hypothesis import given, strategies as st
from pydantic import ValidationError

from dlkit.tools.config.mlflow_settings import (
    MLflowSettings,
    MLflowServerSettings,
    MLflowClientSettings,
)


@pytest.fixture
def mlflow_server_data(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Sample dataflow for MLflowServerSettings testing.

    Returns:
        Dict[str, Any]: MLflow server configuration
    """
    artifacts_dir = tmp_path_factory.mktemp("mlflow_artifacts") / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return {
        "scheme": "https",
        "host": "mlflow.example.com",
        "port": 443,
        "backend_store_uri": "sqlite:///mlflow.db",
        "artifacts_destination": artifacts_dir.resolve().as_uri(),
        "num_workers": 8,
        "keep_alive_interval": 10,
        "shutdown_timeout": 30,
    }


@pytest.fixture
def mlflow_client_data() -> dict[str, Any]:
    """Sample dataflow for MLflowClientSettings testing.

    Returns:
        Dict[str, Any]: MLflow client configuration
    """
    return {
        "experiment_name": "TestExperiment",
        "run_name": "test_run_001",
        "tracking_uri": "http://localhost:5000",
        "register_model": True,
        "registered_model_aliases": ("dataset_A_latest",),
        "registered_model_version_tags": {"team": "platform"},
        "max_trials": 5,
    }


@pytest.fixture
def mlflow_settings_data(
    mlflow_server_data: dict[str, Any], mlflow_client_data: dict[str, Any]
) -> dict[str, Any]:
    """Sample dataflow for complete MLflowSettings testing.

    Args:
        mlflow_server_data: MLflow server configuration
        mlflow_client_data: MLflow client configuration

    Returns:
        Dict[str, Any]: Complete MLflow configuration
    """
    return {"enabled": True, "server": mlflow_server_data, "client": mlflow_client_data}


class TestMLflowServerSettings:
    """Test suite for MLflowServerSettings functionality."""

    def test_initialization_with_defaults(self) -> None:
        """Test MLflowServerSettings initialization with default values."""
        settings = MLflowServerSettings()

        assert settings.scheme == "http"
        assert settings.host == "127.0.0.1"
        assert settings.port == 5000
        assert settings.num_workers == 1  # Default is 1 for SQLite compatibility
        assert settings.keep_alive_interval == 5
        assert settings.shutdown_timeout == 10

    def test_initialization_with_custom_data(self, mlflow_server_data: dict[str, Any]) -> None:
        """Test MLflowServerSettings initialization with custom

        Args:
            mlflow_server_data: MLflow server dataflow fixture
        """
        settings = MLflowServerSettings(**mlflow_server_data)

        assert settings.scheme == "https"
        assert settings.host == "mlflow.example.com"
        assert settings.port == 443
        assert settings.num_workers == 8

    def test_port_validation_valid_range(self) -> None:
        """Test port validation accepts valid port numbers."""
        valid_ports = [1, 80, 443, 5000, 8080, 65535]

        for port in valid_ports:
            settings = MLflowServerSettings(port=port)
            assert settings.port == port

    def test_port_validation_invalid_range(self) -> None:
        """Test port validation rejects invalid port numbers."""
        invalid_ports = [0, -1, 65536, 99999]

        for port in invalid_ports:
            with pytest.raises(ValidationError):
                MLflowServerSettings(port=port)

    def test_command_generation_unix_system(self, mlflow_server_data: dict[str, Any]) -> None:
        """Test command generation for Unix systems.

        Args:
            mlflow_server_data: MLflow server dataflow fixture
        """
        settings = MLflowServerSettings(**mlflow_server_data)
        command = settings.command

        assert "mlflow" in command
        assert "server" in command
        assert "--host" in command
        assert str(settings.host) in command
        assert "--port" in command
        assert str(settings.port) in command
        assert "--backend-store-uri" in command
        assert "--artifacts-destination" in command
        if os.name != "nt":
            assert "--uvicorn-opts" in command
            opts_index = command.index("--uvicorn-opts") + 1
            assert opts_index < len(command)
            uvicorn_opts_value = command[opts_index]
            assert f"--workers {settings.num_workers}" in uvicorn_opts_value
            assert f"--timeout-keep-alive {settings.keep_alive_interval}" in uvicorn_opts_value
            assert f"--timeout-graceful-shutdown {settings.shutdown_timeout}" in uvicorn_opts_value

    @given(st.integers(min_value=1, max_value=65535))
    def test_server_property_valid_ports(self, port: int) -> None:
        """Property test: Server settings accept all valid port numbers.

        Args:
            port: Generated valid port number
        """
        settings = MLflowServerSettings(port=port)

        assert settings.port == port
        assert str(port) in settings.command


class TestMLflowClientSettings:
    """Test suite for MLflowClientSettings functionality."""

    def test_initialization_with_defaults(self) -> None:
        """Test MLflowClientSettings initialization with default values."""
        settings = MLflowClientSettings()

        assert settings.experiment_name == "Experiment"
        assert settings.run_name is None
        assert settings.tracking_uri is None
        assert settings.register_model is True
        assert settings.registered_model_aliases is None
        assert settings.registered_model_version_tags is None
        assert settings.max_trials == 3

    def test_initialization_with_custom_data(self, mlflow_client_data: dict[str, Any]) -> None:
        """Test MLflowClientSettings initialization with custom

        Args:
            mlflow_client_data: MLflow client dataflow fixture
        """
        settings = MLflowClientSettings(**mlflow_client_data)

        assert settings.experiment_name == "TestExperiment"
        assert settings.run_name == "test_run_001"
        assert str(settings.tracking_uri) == "http://localhost:5000/"
        assert settings.register_model is True
        assert settings.registered_model_aliases == ("dataset_A_latest",)
        assert settings.registered_model_version_tags == {"team": "platform"}
        assert settings.max_trials == 5

    @given(st.text(min_size=1, max_size=100), st.integers(min_value=1, max_value=10), st.booleans())
    def test_client_property_configuration(
        self, experiment_name: str, max_trials: int, register_model: bool
    ) -> None:
        """Property test: Client settings accept valid configuration values.

        Args:
            experiment_name: Generated experiment name
            max_trials: Generated max trials value
            register_model: Generated register model flag
        """
        settings = MLflowClientSettings(
            experiment_name=experiment_name, max_trials=max_trials, register_model=register_model
        )

        assert settings.experiment_name == experiment_name
        assert settings.max_trials == max_trials
        assert settings.register_model == register_model


class TestMLflowSettings:
    """Test suite for MLflowSettings functionality."""

    def test_initialization_with_defaults(self) -> None:
        """Test MLflowSettings initialization with default values."""
        settings = MLflowSettings()

        assert settings.enabled is False
        assert isinstance(settings.server, MLflowServerSettings)
        assert isinstance(settings.client, MLflowClientSettings)

    def test_initialization_with_complete_data(self, mlflow_settings_data: dict[str, Any]) -> None:
        """Test MLflowSettings initialization with complete

        Args:
            mlflow_settings_data: Complete MLflow settings dataflow fixture
        """
        settings = MLflowSettings(**mlflow_settings_data)

        assert settings.enabled is True
        assert settings.server.scheme == "https"
        assert settings.client.experiment_name == "TestExperiment"

    def test_default_tracking_uri_deferred_when_not_set(self) -> None:
        """Test tracking URI stays None at config time for deferred runtime resolution.

        When no tracking_uri is configured, the system defers resolution to runtime
        where it defaults to a local SQLite store (locations.mlruns_backend_uri()).
        """
        server_config = {"scheme": "https", "host": "custom.mlflow.com", "port": 8080}
        settings = MLflowSettings(
            enabled=True,
            server=server_config,
            client={},  # No tracking_uri specified
        )

        # URI is None at config time — resolved to SQLite default at runtime
        assert settings.client.tracking_uri is None
        assert settings.tracking_uri is None

    def test_tracking_uri_override_preserved(self) -> None:
        """Test explicit tracking URI is preserved over server default."""
        server_config = {"host": "server.example.com", "port": 5000}
        client_config = {"tracking_uri": "http://override.example.com:9000"}

        settings = MLflowSettings(enabled=True, server=server_config, client=client_config)

        assert str(settings.client.tracking_uri) == "http://override.example.com:9000/"

    def test_enabled_property_with_tracking_uri(self) -> None:
        """Test enabled property works correctly with tracking URI configured."""
        settings = MLflowSettings(enabled=True, client={"tracking_uri": "http://localhost:5000"})

        assert settings.enabled is True
        assert str(settings.client.tracking_uri) == "http://localhost:5000/"

    def test_enabled_property_disabled(self) -> None:
        """Test enabled property works when disabled."""
        settings = MLflowSettings(enabled=False, client={"tracking_uri": "http://localhost:5000"})

        assert settings.enabled is False
        assert str(settings.client.tracking_uri) == "http://localhost:5000/"

    def test_enabled_with_validation(self) -> None:
        """Test that MLflow validation works correctly when enabled."""
        # This should work fine - MLflow enabled with proper tracking URI
        settings = MLflowSettings(enabled=True, client={"tracking_uri": "http://localhost:5000"})
        assert settings.enabled is True

        # Test basic enabled functionality - the model validator now handles validation
        assert settings.enabled is True

    def test_experiment_name_property(self, mlflow_settings_data: dict[str, Any]) -> None:
        """Test experiment_name property returns client experiment name.

        Args:
            mlflow_settings_data: MLflow settings dataflow fixture
        """
        settings = MLflowSettings(**mlflow_settings_data)

        assert settings.experiment_name == "TestExperiment"

    def test_run_name_property(self, mlflow_settings_data: dict[str, Any]) -> None:
        """Test run_name property returns client run name.

        Args:
            mlflow_settings_data: MLflow settings dataflow fixture
        """
        settings = MLflowSettings(**mlflow_settings_data)

        assert settings.run_name == "test_run_001"

    def test_run_name_property_none(self) -> None:
        """Test run_name property returns None when not configured."""
        settings = MLflowSettings()

        assert settings.run_name is None

    def test_tracking_uri_property(self, mlflow_settings_data: dict[str, Any]) -> None:
        """Test tracking_uri property returns string URI.

        Args:
            mlflow_settings_data: MLflow settings dataflow fixture
        """
        settings = MLflowSettings(**mlflow_settings_data)

        assert settings.tracking_uri == "http://localhost:5000/"

    @given(st.booleans())
    def test_mlflow_property_enabled_state(self, enabled: bool) -> None:
        """Property test: MLflow enabled state works correctly.

        Args:
            enabled: Whether MLflow should be enabled
        """
        # Always provide tracking URI when enabled to ensure validation passes
        client_config = {"tracking_uri": "http://localhost:5000"} if enabled else {}

        settings = MLflowSettings(enabled=enabled, client=client_config)

        # Test that enabled property reflects the configured state
        assert settings.enabled is enabled
