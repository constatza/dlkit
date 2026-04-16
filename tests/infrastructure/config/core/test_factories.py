"""Tests for core.factories module.

This module tests the factory pattern implementation for SOLID-compliant
object construction from settings.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from hypothesis import given
from hypothesis import strategies as st

from dlkit.infrastructure.config.core.base_settings import ComponentSettings
from dlkit.infrastructure.config.core.context import BuildContext
from dlkit.infrastructure.config.core.factories import (
    ComponentFactory,
    ComponentRegistry,
    DefaultComponentFactory,
    FactoryProvider,
)

from .conftest import MockComponentSettings


# Test target classes for factory testing
class MockTarget:
    """Simple test target class for factory tests."""

    def __init__(self, param1: str, param2: int = 42, **kwargs):
        """Initialize with parameters for testing compatibility.

        Args:
            param1: Required string parameter
            param2: Optional integer parameter with default
            **kwargs: Additional keyword arguments
        """
        self.param1 = param1
        self.param2 = param2
        self.kwargs = kwargs

        # Set all kwargs as attributes for testing
        for key, value in kwargs.items():
            setattr(self, key, value)


class ComplexMockTarget:
    """Complex test target with more parameters."""

    def __init__(
        self,
        name: str,
        value: int,
        enabled: bool = True,
        config: dict[str, Any] | None = None,
        **extra,
    ):
        """Initialize complex target with multiple parameter types.

        Args:
            name: Required name parameter
            value: Required value parameter
            enabled: Optional boolean parameter
            config: Optional configuration dictionary
            **extra: Additional parameters
        """
        self.name = name
        self.value = value
        self.enabled = enabled
        self.config = config or {}
        self.extra = extra


def function_target(param1: str, param2: int, param3: str = "default") -> dict[str, Any]:
    """Function target for factory testing.

    Args:
        param1: Required parameter
        param2: Required parameter
        param3: Optional parameter with default

    Returns:
        Dict containing all parameters
    """
    return {"param1": param1, "param2": param2, "param3": param3}


class CustomComponentFactory(ComponentFactory[MockTarget]):
    """Custom factory implementation for testing."""

    def create(self, settings: ComponentSettings, context: BuildContext) -> MockTarget:
        """Create component with custom logic.

        Args:
            settings: Component settings
            context: Build context

        Returns:
            MockTarget: Created test target
        """
        typed_settings = settings
        assert isinstance(typed_settings, MockComponentSettings)
        return MockTarget(
            param1=f"custom_{typed_settings.param1}",
            param2=typed_settings.param2 * 2,
            custom_field="added_by_factory",
        )


class TestDefaultComponentFactory:
    """Test suite for DefaultComponentFactory functionality."""

    def test_create_with_class_target(self, sample_build_context: BuildContext) -> None:
        """Test factory creates objects from class references.

        Args:
            sample_build_context: Sample build context fixture
        """
        factory = DefaultComponentFactory()
        settings = MockComponentSettings(name=MockTarget, param1="test_value", param2=100)

        result = factory.create(settings, sample_build_context)

        assert isinstance(result, MockTarget)
        assert result.param1 == "test_value"
        assert result.param2 == 100

    def test_create_with_string_target(self, sample_build_context: BuildContext) -> None:
        """Test factory creates objects from string class names.

        Args:
            sample_build_context: Sample build context fixture
        """
        factory = DefaultComponentFactory()
        settings = MockComponentSettings(
            name="MockTarget",
            module_path="tests.infrastructure.config.core.test_factories",
            param1="string_test",
            param2=200,
        )

        with patch("dlkit.infrastructure.config.core.factories.import_object") as mock_import:
            mock_import.return_value = MockTarget

            result = factory.create(settings, sample_build_context)

            assert isinstance(result, MockTarget)
            assert result.param1 == "string_test"
            assert result.param2 == 200
            mock_import.assert_called_once_with(
                "MockTarget", fallback_module="tests.infrastructure.config.core.test_factories"
            )

    def test_create_with_function_target(self, sample_build_context: BuildContext) -> None:
        """Test factory creates objects from callable targets.

        Args:
            sample_build_context: Sample build context fixture
        """
        factory = DefaultComponentFactory()
        settings = MockComponentSettings(name=function_target, param1="func_test", param2=300)

        result = factory.create(settings, sample_build_context)

        # Factory returns callable as-is for runtime invocation
        assert callable(result)
        assert result == function_target

    def test_create_applies_context_overrides(self) -> None:
        """Test factory applies context overrides to initialization."""
        factory = DefaultComponentFactory()
        settings = MockComponentSettings(name=MockTarget, param1="original_value", param2=100)

        context = BuildContext(
            mode="test",
            overrides={
                "param1": "overridden_value",  # Override test_param -> param1
                "param2": 999,  # Override numeric_param -> param2
            },
        )

        result = factory.create(settings, context)

        assert result.param1 == "overridden_value"
        assert result.param2 == 999

    def test_create_filters_incompatible_kwargs(self, sample_build_context: BuildContext) -> None:
        """Test factory filters out incompatible kwargs.

        Args:
            sample_build_context: Sample build context fixture
        """
        factory = DefaultComponentFactory()
        settings = MockComponentSettings.model_validate(
            {
                "name": MockTarget,
                "param1": "filter_test",
                "param2": 400,
                "incompatible_param": "should_be_filtered",
            }
        )

        result = factory.create(settings, sample_build_context)

        assert isinstance(result, MockTarget)
        assert result.param1 == "filter_test"
        assert result.param2 == 400
        assert not hasattr(result, "incompatible_param")

    def test_create_invalid_target_type_raises_error(
        self, sample_build_context: BuildContext
    ) -> None:
        """Test factory raises TypeError for invalid target types.

        Args:
            sample_build_context: Sample build context fixture
        """
        factory = DefaultComponentFactory()
        settings = MockComponentSettings(
            name="not_a_class_or_callable",  # String that won't resolve
            module_path="nonexistent.module",
        )

        with patch("dlkit.infrastructure.config.core.factories.import_object") as mock_import:
            mock_import.return_value = "not_callable"  # Return non-callable object

            with pytest.raises(TypeError, match="must be a class or callable"):
                factory.create(settings, sample_build_context)

    @given(st.text(min_size=1, max_size=50), st.integers(min_value=1, max_value=1000))
    def test_create_property_parameter_compatibility(self, param1: str, param2: int) -> None:
        """Property test: Factory correctly maps compatible parameters.

        Args:
            param1: Generated parameter value
            param2: Generated parameter value
        """
        factory = DefaultComponentFactory()
        settings = MockComponentSettings(name=MockTarget, param1=param1, param2=param2)

        context = BuildContext(mode="test")
        result = factory.create(settings, context)

        assert isinstance(result, MockTarget)
        assert result.param1 == param1
        assert result.param2 == param2


class TestComponentRegistry:
    """Test suite for ComponentRegistry functionality."""

    def test_registry_initialization(self) -> None:
        """Test ComponentRegistry initializes with default factory."""
        registry = ComponentRegistry()

        assert isinstance(registry._default_factory, DefaultComponentFactory)
        assert len(registry._factories) == 0

    def test_register_custom_factory(self) -> None:
        """Test registry can register custom factories."""
        registry = ComponentRegistry()
        custom_factory = CustomComponentFactory()

        registry.register_factory(MockComponentSettings, custom_factory)

        assert registry._factories[MockComponentSettings] == custom_factory

    def test_get_factory_returns_custom_when_registered(self) -> None:
        """Test get_factory returns custom factory when registered."""
        registry = ComponentRegistry()
        custom_factory = CustomComponentFactory()
        registry.register_factory(MockComponentSettings, custom_factory)

        result = registry.get_factory(MockComponentSettings)

        assert result == custom_factory

    def test_get_factory_returns_default_when_not_registered(self) -> None:
        """Test get_factory returns default factory when no custom registered."""
        registry = ComponentRegistry()

        result = registry.get_factory(MockComponentSettings)

        assert isinstance(result, DefaultComponentFactory)

    def test_create_component_uses_appropriate_factory(
        self, sample_build_context: BuildContext
    ) -> None:
        """Test create_component uses the correct factory.

        Args:
            sample_build_context: Sample build context fixture
        """
        registry = ComponentRegistry()
        custom_factory = CustomComponentFactory()
        registry.register_factory(MockComponentSettings, custom_factory)

        settings = MockComponentSettings(name=MockTarget, param1="registry_test", param2=50)

        result = registry.create_component(settings, sample_build_context)

        # Custom factory modifies the creation process
        assert isinstance(result, MockTarget)
        assert result.param1 == "custom_registry_test"  # Modified by custom factory
        assert result.param2 == 100  # Doubled by custom factory
        assert hasattr(result, "custom_field")


class TestFactoryProvider:
    """Test suite for FactoryProvider singleton functionality."""

    def test_get_registry_returns_singleton(self) -> None:
        """Test get_registry returns the same instance (singleton pattern)."""
        registry1 = FactoryProvider.get_registry()
        registry2 = FactoryProvider.get_registry()

        assert registry1 is registry2
        assert isinstance(registry1, ComponentRegistry)

    def test_create_component_delegates_to_registry(
        self, sample_build_context: BuildContext
    ) -> None:
        """Test create_component delegates to the singleton registry.

        Args:
            sample_build_context: Sample build context fixture
        """
        settings = MockComponentSettings(name=MockTarget, param1="provider_test", param2=75)

        result = FactoryProvider.create_component(settings, sample_build_context)

        assert isinstance(result, MockTarget)
        assert result.param1 == "provider_test"
        assert result.param2 == 75

    def test_register_factory_delegates_to_registry(self) -> None:
        """Test register_factory delegates to the singleton registry."""
        custom_factory = CustomComponentFactory()

        FactoryProvider.register_factory(MockComponentSettings, custom_factory)

        # Verify registration worked by checking the registry directly
        registry = FactoryProvider.get_registry()
        assert registry.get_factory(MockComponentSettings) == custom_factory

    def test_global_registry_persistence(self, sample_build_context: BuildContext) -> None:
        """Test that registrations persist across FactoryProvider calls."""
        # Register a custom factory
        custom_factory = CustomComponentFactory()
        FactoryProvider.register_factory(MockComponentSettings, custom_factory)

        # Use the factory through FactoryProvider
        settings = MockComponentSettings(name=MockTarget, param1="persistence_test", param2=25)

        result = FactoryProvider.create_component(settings, sample_build_context)

        # Verify custom factory was used
        assert result.param1 == "custom_persistence_test"
        assert result.param2 == 50
        assert hasattr(result, "custom_field")
