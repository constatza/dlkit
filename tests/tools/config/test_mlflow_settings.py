"""Tests for flat MLflow settings."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, strategies as st
from pydantic import ValidationError

from dlkit.tools.config.mlflow_settings import MLflowSettings


@pytest.fixture
def mlflow_settings_data() -> dict[str, Any]:
    """Sample data for MLflowSettings testing."""
    return {
        "experiment_name": "TestExperiment",
        "run_name": "test_run_001",
        "register_model": True,
        "registered_model_name": "RegisteredFFNN",
        "registered_model_aliases": ("dataset_A_latest",),
        "registered_model_version_tags": {"team": "platform"},
        "max_retries": 5,
    }


class TestMLflowSettings:
    """Test suite for flattened MLflowSettings functionality."""

    def test_initialization_with_defaults(self) -> None:
        settings = MLflowSettings()

        assert settings.experiment_name == "Experiment"
        assert settings.run_name is None
        assert settings.register_model is False
        assert settings.registered_model_name is None
        assert settings.registered_model_aliases is None
        assert settings.registered_model_version_tags is None
        assert settings.max_retries == 3

    def test_initialization_with_custom_data(self, mlflow_settings_data: dict[str, Any]) -> None:
        settings = MLflowSettings(**mlflow_settings_data)

        assert settings.experiment_name == "TestExperiment"
        assert settings.run_name == "test_run_001"
        assert settings.register_model is True
        assert settings.registered_model_name == "RegisteredFFNN"
        assert settings.registered_model_aliases == ("dataset_A_latest",)
        assert settings.registered_model_version_tags == {"team": "platform"}
        assert settings.max_retries == 5

    @given(
        st.text(min_size=1, max_size=80),
        st.integers(min_value=1, max_value=10),
        st.booleans(),
    )
    def test_property_based_configuration(
        self,
        experiment_name: str,
        max_retries: int,
        register_model: bool,
    ) -> None:
        settings = MLflowSettings(
            experiment_name=experiment_name,
            max_retries=max_retries,
            register_model=register_model,
        )

        assert settings.experiment_name == experiment_name
        assert settings.max_retries == max_retries
        assert settings.register_model is register_model

    def test_legacy_nested_sections_fail_with_migration_message(self) -> None:
        with pytest.raises(ValidationError, match="Legacy MLflow config sections"):
            MLflowSettings(client={"experiment_name": "exp"})  # type: ignore[arg-type]

        with pytest.raises(ValidationError, match="Legacy MLflow config sections"):
            MLflowSettings(server={"host": "127.0.0.1"})  # type: ignore[arg-type]

    def test_infra_fields_in_toml_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="env-only"):
            MLflowSettings(tracking_uri="http://127.0.0.1:5000")  # type: ignore[arg-type]

        with pytest.raises(ValidationError, match="env-only"):
            MLflowSettings(artifacts_destination="file:///C:/artifacts")  # type: ignore[arg-type]

    def test_enabled_field_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="no longer has an 'enabled' field"):
            MLflowSettings(enabled=True)  # type: ignore[arg-type]
