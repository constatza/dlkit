"""Simple tests for core settings functionality without complex imports."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

# Direct imports to avoid complex dependency chain
from dlkit.tools.config.core.base_settings import (
    BasicSettings,
    ComponentSettings,
    HyperParameterSettings,
)
from dlkit.tools.config.core.context import BuildContext


class MockBasicSettings(BasicSettings):
    """Test implementation of BasicSettings."""

    name: str
    value: int
    enabled: bool = True


class MockComponentSettings(ComponentSettings):
    """Test implementation of ComponentSettings."""

    test_param: str = "default"
    numeric_param: int = 42


class MockHyperParameterSettings(HyperParameterSettings):
    """Test implementation of HyperParameterSettings."""

    learning_rate: float | dict[str, Any] = 0.001
    batch_size: int | dict[str, Any] = 32


class MockTrial:
    """Mock Optuna trial for testing."""

    def __init__(self):
        self.suggestions = {}

    def suggest_int(self, name: str, low: int, high: int, step: int = 1, log: bool = False) -> int:
        value = (low + high) // 2
        self.suggestions[name] = value
        return value

    def suggest_float(
        self, name: str, low: float, high: float, step: float | None = None, log: bool = False
    ) -> float:
        value = (low + high) / 2
        self.suggestions[name] = value
        return value

    def suggest_categorical(self, name: str, choices: tuple) -> Any:
        value = choices[0]
        self.suggestions[name] = value
        return value


class MockBasicSettingsFunctionality:
    """Basic tests for settings functionality."""

    def test_basic_settings_creation(self):
        """Test basic settings can be created and are frozen."""
        settings = MockBasicSettings(name="test", value=100)

        assert settings.name == "test"
        assert settings.value == 100
        assert settings.enabled is True

        # Should be frozen
        with pytest.raises(ValidationError):
            settings.name = "modified"

    def test_component_settings_creation(self):
        """Test component settings can be created."""
        settings = MockComponentSettings(
            name="TestComponent", test_param="custom", numeric_param=999
        )

        assert settings.name == "TestComponent"
        assert settings.test_param == "custom"
        assert settings.numeric_param == 999

    def test_component_settings_get_init_kwargs(self):
        """Test component settings get_init_kwargs excludes component fields."""
        settings = MockComponentSettings(
            name="TestComponent", module_path="test.module", test_param="test", numeric_param=123
        )

        kwargs = settings.get_init_kwargs()

        assert "name" not in kwargs
        assert "module_path" not in kwargs
        assert "test_param" in kwargs
        assert "numeric_param" in kwargs

    # Legacy direct sampling tests removed (moved to samplers).


class TestBuildContextFunctionality:
    """Basic tests for BuildContext functionality."""

    def test_build_context_creation(self):
        """Test BuildContext can be created with defaults."""
        context = BuildContext(mode="testing")

        assert context.mode == "testing"
        assert context.device == "auto"
        assert context.random_seed is None
        assert context.overrides == {}

    def test_build_context_with_overrides(self):
        """Test BuildContext with_overrides method."""
        original = BuildContext(mode="training", overrides={"param1": "value1"})

        updated = original.with_overrides(param2="value2", param1="new_value1")

        # Original unchanged
        assert original.overrides == {"param1": "value1"}

        # Updated has merged overrides
        assert updated.overrides == {"param1": "new_value1", "param2": "value2"}

    def test_build_context_get_override(self):
        """Test BuildContext get_override method."""
        context = BuildContext(mode="testing", overrides={"key1": "value1", "key2": 42})

        assert context.get_override("key1") == "value1"
        assert context.get_override("key2") == 42
        assert context.get_override("missing") is None
        assert context.get_override("missing", "default") == "default"
