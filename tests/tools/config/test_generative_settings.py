"""Tests for generative algorithm settings configuration.

This module tests the discriminated union architecture for generative algorithms:
- FlowMatchingSettings: Flow matching with velocity field training
- CNFSettings: Continuous normalising flows with ODE integration
- GenerativeSettings union: Discriminated union routing to correct class
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import ValidationError

from dlkit.infrastructure.config.general_settings import GeneralSettings
from dlkit.infrastructure.config.generative_settings import (
    CNFSettings,
    FlowMatchingSettings,
    GenerativeSettings,
)


class TestFlowMatchingSettings:
    """Test suite for FlowMatchingSettings."""

    def test_flow_matching_defaults(self) -> None:
        """Test FlowMatchingSettings with default values."""
        settings = FlowMatchingSettings()

        assert settings.algorithm == "flow_matching"
        assert settings.path_type == "linear"
        assert settings.target_type == "displacement"
        assert settings.solver == "euler"
        assert settings.n_inference_steps == 100
        assert settings.val_seed == 42
        assert settings.x1_key == "x1"

    def test_flow_matching_custom_values(self) -> None:
        """Test FlowMatchingSettings with custom values."""
        settings = FlowMatchingSettings(
            path_type="noise_schedule",
            solver="heun",
            n_inference_steps=50,
            val_seed=123,
            x1_key="target",
        )

        assert settings.path_type == "noise_schedule"
        assert settings.solver == "heun"
        assert settings.n_inference_steps == 50
        assert settings.val_seed == 123
        assert settings.x1_key == "target"

    def test_flow_matching_invalid_path_type(self) -> None:
        """Test FlowMatchingSettings rejects invalid path_type."""
        with pytest.raises(ValidationError):
            FlowMatchingSettings(path_type=cast(Any, "invalid"))

    def test_flow_matching_invalid_solver(self) -> None:
        """Test FlowMatchingSettings rejects invalid solver."""
        with pytest.raises(ValidationError):
            FlowMatchingSettings(solver=cast(Any, "rk45"))

    def test_flow_matching_serialization(self) -> None:
        """Test FlowMatchingSettings serialization and deserialization."""
        original = FlowMatchingSettings(
            path_type="noise_schedule",
            solver="heun",
            n_inference_steps=75,
        )

        # Serialize to dict
        data = original.model_dump()

        # Deserialize from dict
        restored = FlowMatchingSettings(**data)

        assert restored.algorithm == original.algorithm
        assert restored.path_type == original.path_type
        assert restored.solver == original.solver
        assert restored.n_inference_steps == original.n_inference_steps


class TestCNFSettings:
    """Test suite for CNFSettings."""

    def test_cnf_defaults(self) -> None:
        """Test CNFSettings with default values."""
        settings = CNFSettings()

        assert settings.algorithm == "cnf"
        assert settings.solver == "euler"
        assert settings.n_inference_steps == 100
        assert settings.divergence == "hutchinson"
        assert settings.val_seed == 42

    def test_cnf_custom_values(self) -> None:
        """Test CNFSettings with custom values."""
        settings = CNFSettings(
            solver="heun",
            n_inference_steps=200,
            divergence="exact",
            val_seed=999,
        )

        assert settings.solver == "heun"
        assert settings.n_inference_steps == 200
        assert settings.divergence == "exact"
        assert settings.val_seed == 999

    def test_cnf_invalid_solver(self) -> None:
        """Test CNFSettings rejects invalid solver."""
        with pytest.raises(ValidationError):
            CNFSettings(solver=cast(Any, "invalid"))

    def test_cnf_invalid_divergence(self) -> None:
        """Test CNFSettings rejects invalid divergence."""
        with pytest.raises(ValidationError):
            CNFSettings(divergence=cast(Any, "trace"))

    def test_cnf_serialization(self) -> None:
        """Test CNFSettings serialization and deserialization."""
        original = CNFSettings(
            solver="heun",
            divergence="exact",
            n_inference_steps=150,
        )

        data = original.model_dump()
        restored = CNFSettings(**data)

        assert restored.algorithm == original.algorithm
        assert restored.solver == original.solver
        assert restored.divergence == original.divergence
        assert restored.n_inference_steps == original.n_inference_steps


class TestGenerativeSettings:
    """Test suite for GenerativeSettings union type."""

    def test_generative_settings_flow_matching(self) -> None:
        """Test GenerativeSettings correctly routes to FlowMatchingSettings."""
        data: dict[str, Any] = {
            "algorithm": "flow_matching",
            "path_type": "linear",
            "solver": "euler",
        }

        # Create through union (simulating TOML parsing)
        settings = FlowMatchingSettings(**data)

        assert isinstance(settings, FlowMatchingSettings)
        assert settings.algorithm == "flow_matching"

    def test_generative_settings_cnf(self) -> None:
        """Test GenerativeSettings correctly routes to CNFSettings."""
        data: dict[str, Any] = {
            "algorithm": "cnf",
            "solver": "heun",
            "divergence": "exact",
        }

        settings = CNFSettings(**data)

        assert isinstance(settings, CNFSettings)
        assert settings.algorithm == "cnf"

    def test_generative_settings_invalid_algorithm(self) -> None:
        """Test GenerativeSettings rejects invalid algorithm."""
        # Note: Union discriminator requires one of the valid types
        with pytest.raises((ValidationError, TypeError)):
            data: dict[str, Any] = {
                "algorithm": "unknown_algorithm",
            }
            # Attempting to create with invalid discriminator should fail
            from pydantic import TypeAdapter

            ta = TypeAdapter(GenerativeSettings)
            ta.validate_python(data)


class TestGeneralSettingsIntegration:
    """Test suite for GeneralSettings with GENERATIVE field."""

    def test_general_settings_with_flow_matching(self) -> None:
        """Test GeneralSettings can include FlowMatchingSettings."""
        gen_settings = FlowMatchingSettings(
            path_type="noise_schedule",
            solver="heun",
        )

        general = GeneralSettings(GENERATIVE=gen_settings)

        assert general.GENERATIVE is not None
        assert isinstance(general.GENERATIVE, FlowMatchingSettings)
        assert general.GENERATIVE.algorithm == "flow_matching"
        assert general.GENERATIVE.path_type == "noise_schedule"

    def test_general_settings_with_cnf(self) -> None:
        """Test GeneralSettings can include CNFSettings."""
        gen_settings = CNFSettings(
            divergence="exact",
            n_inference_steps=150,
        )

        general = GeneralSettings(GENERATIVE=gen_settings)

        assert general.GENERATIVE is not None
        assert isinstance(general.GENERATIVE, CNFSettings)
        assert general.GENERATIVE.algorithm == "cnf"
        assert general.GENERATIVE.divergence == "exact"

    def test_general_settings_without_generative(self) -> None:
        """Test GeneralSettings with no generative config (default)."""
        general = GeneralSettings()

        assert general.GENERATIVE is None

    def test_general_settings_generative_serialization(self) -> None:
        """Test GeneralSettings serialization with generative config."""
        gen_settings = FlowMatchingSettings(x1_key="targets")
        general = GeneralSettings(GENERATIVE=gen_settings)

        data = general.model_dump()

        assert data["GENERATIVE"] is not None
        assert data["GENERATIVE"]["algorithm"] == "flow_matching"
        assert data["GENERATIVE"]["x1_key"] == "targets"

    def test_general_settings_generative_roundtrip(self) -> None:
        """Test GeneralSettings serialization roundtrip with GENERATIVE."""
        original = GeneralSettings(
            GENERATIVE=FlowMatchingSettings(
                path_type="noise_schedule",
                solver="heun",
            )
        )

        data = original.model_dump()
        restored = GeneralSettings(**data)

        generative = restored.GENERATIVE
        assert isinstance(generative, FlowMatchingSettings)
        assert generative.path_type == "noise_schedule"
        assert generative.solver == "heun"


class TestGenerativeSettingsDocstring:
    """Test that generative settings have proper documentation."""

    def test_flow_matching_has_docstring(self) -> None:
        """Test FlowMatchingSettings has docstring."""
        assert FlowMatchingSettings.__doc__ is not None
        assert "flow matching" in FlowMatchingSettings.__doc__.lower()

    def test_cnf_has_docstring(self) -> None:
        """Test CNFSettings has docstring."""
        assert CNFSettings.__doc__ is not None
        assert "continuous" in CNFSettings.__doc__.lower()

    def test_generative_settings_usage_example(self) -> None:
        """Test that module docstring includes usage example."""
        from dlkit.infrastructure.config import generative_settings as mod

        assert mod.__doc__ is not None
        assert "TOML" in mod.__doc__
        assert "algorithm" in mod.__doc__
