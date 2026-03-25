"""Tests for core.base_settings module.

This module tests the foundational settings classes following SOLID principles:
- BasicSettings: Base configuration with validation
- ComponentSettings: Dynamic component configuration
- HyperParameterSettings: Hyperparameter optimization support
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from .conftest import MockBasicSettings, MockComponentSettings, valid_name


class TestBasicSettings:
    """Test suite for BasicSettings functionality."""

    def test_basic_settings_initialization(self, basic_settings_data: dict[str, Any]) -> None:
        """Test BasicSettings can be initialized with valid

        Args:
            basic_settings_data: Valid settings dataflow fixture
        """
        settings = MockBasicSettings(**basic_settings_data)

        assert settings.name == basic_settings_data["name"]
        assert settings.value == basic_settings_data["value"]
        assert settings.enabled == basic_settings_data["enabled"]

    def test_basic_settings_validation_on_creation(self) -> None:
        """Test BasicSettings validates dataflow on creation."""
        with pytest.raises(ValidationError):
            MockBasicSettings.model_validate(
                {"name": "", "value": "invalid_int", "enabled": "not_bool"}
            )

    # Removed to_dict_compatible_with tests; factories handle construction explicitly.

    @given(st.data())
    def test_basic_settings_property_serialization_roundtrip(self, data) -> None:
        """Property test: BasicSettings serialization roundtrip preserves

        Args:
            data: Hypothesis dataflow source
        """
        # Generate valid test dataflow
        name = data.draw(st.text(min_size=1, max_size=100))
        value = data.draw(st.integers())
        enabled = data.draw(st.booleans())

        original = MockBasicSettings(name=name, value=value, enabled=enabled)
        serialized = original.model_dump()
        reconstructed = MockBasicSettings(**serialized)

        assert original == reconstructed


class TestComponentSettings:
    """Test suite for ComponentSettings functionality."""

    def test_component_settings_initialization(
        self, component_settings_data: dict[str, Any]
    ) -> None:
        """Test ComponentSettings initialization with valid

        Args:
            component_settings_data: Valid component settings dataflow fixture
        """
        settings = MockComponentSettings(**component_settings_data)

        assert settings.name == component_settings_data["name"]
        assert settings.module_path == component_settings_data["module_path"]
        assert settings.param1 == component_settings_data["param1"]
        assert settings.param2 == component_settings_data["param2"]

    def test_component_settings_allows_extra_fields(self) -> None:
        """Test ComponentSettings allows extra fields (extra='allow')."""
        settings = MockComponentSettings.model_validate(
            {
                "name": "TestComponent",
                "module_path": "test.module",
                "extra_field": "extra_value",
                "another_extra": 123,
            }
        )
        extra = settings.model_extra or {}

        assert settings.name == "TestComponent"
        assert extra["extra_field"] == "extra_value"
        assert extra["another_extra"] == 123

    def test_get_init_kwargs_excludes_component_fields(
        self, component_settings_data: dict[str, Any]
    ) -> None:
        """Test get_init_kwargs excludes component-specific fields.

        Args:
            component_settings_data: Valid component settings dataflow fixture
        """
        settings = MockComponentSettings(**component_settings_data)
        kwargs = settings.get_init_kwargs()

        assert "name" not in kwargs
        assert "module_path" not in kwargs
        assert "param1" in kwargs
        assert "param2" in kwargs

    def test_get_init_kwargs_custom_exclude(self, component_settings_data: dict[str, Any]) -> None:
        """Test get_init_kwargs respects custom exclude set.

        Args:
            component_settings_data: Valid component settings dataflow fixture
        """
        settings = MockComponentSettings(**component_settings_data)
        kwargs = settings.get_init_kwargs(exclude={"param1"})

        assert "name" not in kwargs
        assert "module_path" not in kwargs
        assert "param1" not in kwargs
        assert "param2" in kwargs

    @given(valid_name(), st.text(min_size=1, max_size=50))  # ty: ignore[missing-argument]
    def test_component_settings_property_valid_names(self, name: str, module_path: str) -> None:
        """Property test: ComponentSettings accepts valid component names.

        Args:
            name: Generated component name (mapped to 'name' field)
            module_path: Generated module path
        """
        settings = MockComponentSettings(name=name, module_path=module_path)

        assert settings.name == name
        assert settings.module_path == module_path


# Legacy HyperParameterSettings tests removed (sampling moved to samplers).
