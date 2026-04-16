"""Integration tests for settings module.

This module tests the integration between different settings components,
factory pattern usage, and end-to-end workflows following SOLID principles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from dlkit.infrastructure.config import (
    BuildContext,
    ComponentFactory,
    FactoryProvider,
    GeneralSettings,
)
from dlkit.infrastructure.config.core.base_settings import ComponentSettings
from dlkit.infrastructure.config.core.factories import DefaultComponentFactory
from dlkit.infrastructure.config.model_components import ModelComponentSettings


def _expect_not_none[T](value: T | None) -> T:
    assert value is not None
    return value


# Mock components for integration testing


@pytest.fixture(autouse=True)
def _isolate_factory_provider_registry():
    """Prevent global FactoryProvider state from leaking across test modules."""
    previous = FactoryProvider.reset_for_testing()
    yield
    FactoryProvider.restore_for_testing(previous)


class MockModel:
    """Mock model for integration testing."""

    def __init__(self, input_size: int = 100, output_size: int = 10, **kwargs):
        """Initialize mock model.

        Args:
            input_size: Input dimension size
            output_size: Output dimension size
            **kwargs: Additional parameters
        """
        self.input_size = input_size
        self.output_size = output_size
        self.kwargs = kwargs

        # Set additional attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockDataModule:
    """Mock datamodule for integration testing."""

    def __init__(self, batch_size: int = 32, num_workers: int = 4, **kwargs):
        """Initialize mock datamodule.

        Args:
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes
            **kwargs: Additional parameters
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs


class MockMetric:
    """Mock metric for integration testing."""

    def __init__(self, name: str = "accuracy", **kwargs):
        """Initialize mock metric.

        Args:
            name: Metric name
            **kwargs: Additional parameters
        """
        self.name = name
        self.kwargs = kwargs

        # Set additional attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class CustomModelFactory(ComponentFactory[MockModel]):
    """Custom factory for mock models."""

    def create(self, settings: ComponentSettings, context: BuildContext) -> MockModel:
        """Create mock model with custom logic.

        Args:
            settings: Model component settings
            context: Build context

        Returns:
            MockModel: Created mock model
        """
        kwargs = settings.get_init_kwargs()
        # Add context-specific customizations
        if context.mode == "training":
            kwargs["training_mode"] = True
        elif context.mode == "inference":
            kwargs["inference_mode"] = True

        return MockModel(**kwargs)


@pytest.fixture
def complete_config_data() -> dict[str, Any]:
    """Complete configuration for integration testing.

    Returns:
        Dict[str, Any]: Complete configuration dataflow
    """
    return {
        "SESSION": {
            "name": "integration_session",
            "inference": False,  # Training mode
            "seed": 42,
        },
        "MODEL": {
            "name": "IntegrationModel",
            "module_path": "test.models",
            "input_size": 128,
            "output_size": 10,
        },
        "MLFLOW": {"experiment_name": "integration_experiment"},
        "OPTUNA": {"enabled": True, "n_trials": 10},
        "DATAMODULE": {
            "name": "IntegrationDataModule",
            "module_path": "test.datamodules",
            "dataloader": {
                "batch_size": 64,
                "num_workers": 8,
            },
        },
        "TRAINING": {"epochs": 50, "trainer": {"accelerator": "cpu"}},
    }


@pytest.fixture
def integration_config_file(tmp_path: Path, complete_config_data: dict[str, Any]) -> Path:
    """Create integration test configuration file.

    Args:
        tmp_path: Pytest temporary path fixture
        complete_config_data: Complete configuration dataflow

    Returns:
        Path: Path to integration config file
    """
    config_content = """
[SESSION]
name = "integration_session"
inference = false
seed = 42

[MODEL]
name = "IntegrationModel"
module_path = "test.models"
input_size = 128
output_size = 10

[MLFLOW]
experiment_name = "integration_experiment"

[OPTUNA]
enabled = true
n_trials = 10

[DATAMODULE]
name = "IntegrationDataModule"
module_path = "test.datamodules"

[DATAMODULE.dataloader]
batch_size = 64
num_workers = 8

[TRAINING]
epochs = 50

[TRAINING.trainer]
accelerator = "cpu"
"""

    config_file = tmp_path / "integration_config.toml"
    config_file.write_text(config_content)
    return config_file


class TestSettingsFactoryIntegration:
    """Test integration between settings and factory pattern."""

    def test_default_factory_component_creation(self, sample_build_context: BuildContext) -> None:
        """Test default factory creates components from settings.

        Args:
            sample_build_context: Build context fixture
        """
        settings = ModelComponentSettings.model_validate(
            {
                "name": "MockModel",
                "module_path": "tests.infrastructure.config.test_integration",
                "input_size": 256,
                "output_size": 20,
            }
        )

        factory = DefaultComponentFactory()
        result = factory.create(settings, sample_build_context)

        assert isinstance(result, MockModel)
        assert result.input_size == 256
        assert result.output_size == 20

    def test_custom_factory_registration_and_usage(
        self, sample_build_context: BuildContext
    ) -> None:
        """Test custom factory registration and usage through FactoryProvider.

        Args:
            sample_build_context: Build context fixture
        """
        # Register custom factory
        custom_factory = CustomModelFactory()
        FactoryProvider.register_factory(ModelComponentSettings, custom_factory)

        settings = ModelComponentSettings.model_validate(
            {
                "name": "MockModel",
                "module_path": "tests.infrastructure.config.test_integration",
                "input_size": 100,
                "output_size": 5,
            }
        )

        # Create component through FactoryProvider
        result = FactoryProvider.create_component(settings, sample_build_context)

        assert isinstance(result, MockModel)
        assert result.input_size == 100
        assert hasattr(result, "training_mode")  # Added by custom factory
        assert result.training_mode is True

    def test_factory_context_override_application(self) -> None:
        """Test factory applies build context overrides correctly."""
        settings = ModelComponentSettings.model_validate(
            {
                "name": "MockModel",
                "module_path": "tests.infrastructure.config.test_integration",
                "input_size": 128,
                "output_size": 10,
            }
        )

        context = BuildContext(
            mode="inference",
            overrides={
                "input_size": 512,  # Override original value
                "custom_param": "context_value",
            },
        )

        factory = DefaultComponentFactory()
        result = factory.create(settings, context)

        assert result.input_size == 512  # Override applied
        assert result.custom_param == "context_value"  # Context value added

    @patch("dlkit.infrastructure.registry.resolve.import_object")
    def test_factory_string_import_integration(
        self, mock_import, sample_build_context: BuildContext
    ) -> None:
        """Test factory handles string imports correctly.

        Args:
            mock_import: Mock import_object function
            sample_build_context: Build context fixture
        """
        mock_import.return_value = MockModel

        settings = ModelComponentSettings.model_validate(
            {
                "name": "MockModel",
                "module_path": "test.models.mock",
                "input_size": 64,
            }
        )

        factory = DefaultComponentFactory()
        result = factory.create(settings, sample_build_context)

        assert isinstance(result, MockModel)
        assert result.input_size == 64
        mock_import.assert_called_once_with("MockModel", fallback_module="test.models.mock")


class TestGeneralSettingsEndToEndIntegration:
    """Test end-to-end integration through GeneralSettings."""

    def test_settings_loading_and_validation_integration(
        self, integration_config_file: Path
    ) -> None:
        """Test complete settings loading and validation flow.

        Args:
            integration_config_file: Integration config file fixture
        """
        from dlkit.infrastructure.config import load_settings

        settings = load_settings(integration_config_file)
        session = _expect_not_none(settings.SESSION)
        model = _expect_not_none(settings.MODEL)
        optuna = _expect_not_none(settings.OPTUNA)
        datamodule = _expect_not_none(settings.DATAMODULE)

        # Verify all components are properly loaded
        assert session.name == "integration_session"
        assert (model.model_extra or {})["input_size"] == 128
        assert settings.MLFLOW is not None
        assert settings.MLFLOW.experiment_name == "integration_experiment"  # Configured value
        assert optuna.enabled is True
        assert datamodule.dataloader.batch_size == 64

    def test_settings_mode_specific_configuration_access(
        self, complete_config_data: dict[str, Any]
    ) -> None:
        """Test mode-specific configuration access integration.

        Args:
            complete_config_data: Complete configuration dataflow fixture
        """
        settings = GeneralSettings.model_validate(complete_config_data)

        # Test mode detection
        assert settings.is_training is True
        assert settings.is_inference is False

        # Test feature flags
        assert settings.mlflow_enabled is True
        assert settings.optuna_enabled is True
        assert settings.has_training_config is True
        assert settings.has_data_config is True

        # Test configuration access
        training_config = settings.get_training_config()
        assert training_config.epochs == 50

        datamodule_config = settings.get_datamodule_config()
        assert datamodule_config.dataloader.batch_size == 64

    # Legacy hyperparameter sampling test removed (sampling moved to samplers).

    def test_settings_validation_error_integration(self) -> None:
        """Test validation error integration across settings hierarchy."""
        # Test inference mode without checkpoint
        invalid_config = {
            "SESSION": {"inference": True, "name": "test_session"},
            "MODEL": {
                "name": "TestModel"
                # Missing checkpoint for inference mode
            },
        }

        with pytest.raises(ValueError, match="Checkpoint path must be provided"):
            GeneralSettings.model_validate(invalid_config)


class TestFactoryProviderSingletonIntegration:
    """Test FactoryProvider singleton behavior in integration scenarios."""

    def test_factory_provider_global_state_persistence(
        self, sample_build_context: BuildContext
    ) -> None:
        """Test FactoryProvider maintains global state across operations.

        Args:
            sample_build_context: Build context fixture
        """
        # Register a factory
        custom_factory = CustomModelFactory()
        FactoryProvider.register_factory(ModelComponentSettings, custom_factory)

        # Create settings and use factory in different contexts
        settings1 = ModelComponentSettings.model_validate(
            {
                "name": "MockModel",
                "module_path": "tests.infrastructure.config.test_integration",
                "input_size": 100,
            }
        )
        settings2 = ModelComponentSettings.model_validate(
            {
                "name": "MockModel",
                "module_path": "tests.infrastructure.config.test_integration",
                "input_size": 200,
            }
        )

        # Both should use the custom factory
        result1 = FactoryProvider.create_component(settings1, sample_build_context)
        result2 = FactoryProvider.create_component(settings2, sample_build_context)

        assert hasattr(result1, "training_mode")  # Custom factory feature
        assert hasattr(result2, "training_mode")  # Custom factory feature
        assert result1.input_size == 100
        assert result2.input_size == 200

    def test_factory_provider_multiple_factory_registration(
        self, sample_build_context: BuildContext
    ) -> None:
        """Test FactoryProvider handles multiple factory registrations.

        Args:
            sample_build_context: Build context fixture
        """
        # Register factories for different settings types
        from dlkit.infrastructure.config.model_components import MetricComponentSettings

        model_factory = CustomModelFactory()

        class CustomMetricFactory(ComponentFactory[MockMetric]):
            def create(self, settings: ComponentSettings, context: BuildContext) -> MockMetric:
                return MockMetric(name="custom_metric")

        metric_factory = CustomMetricFactory()

        FactoryProvider.register_factory(ModelComponentSettings, model_factory)
        FactoryProvider.register_factory(MetricComponentSettings, metric_factory)

        # Test model creation
        model_settings = ModelComponentSettings(
            name="MockModel", module_path="tests.infrastructure.config.test_integration"
        )
        model = FactoryProvider.create_component(model_settings, sample_build_context)
        assert isinstance(model, MockModel)
        assert hasattr(model, "training_mode")

        # Test metric creation
        metric_settings = MetricComponentSettings(name="MockMetric")
        metric = FactoryProvider.create_component(metric_settings, sample_build_context)
        assert isinstance(metric, MockMetric)
        assert metric.name == "custom_metric"


class TestBuildContextIntegration:
    """Test BuildContext integration across factory operations."""

    def test_build_context_mode_specific_behavior(self) -> None:
        """Test BuildContext enables mode-specific behavior in factories."""
        settings = ModelComponentSettings.model_validate(
            {
                "name": "MockModel",
                "module_path": "tests.infrastructure.config.test_integration",
                "input_size": 100,
            }
        )
        custom_factory = CustomModelFactory()

        # Test training mode
        training_context = BuildContext(mode="training")
        training_result = custom_factory.create(settings, training_context)
        assert hasattr(training_result, "training_mode")
        assert training_result.training_mode is True

        # Test inference mode
        inference_context = BuildContext(mode="inference")
        inference_result = custom_factory.create(settings, inference_context)
        assert hasattr(inference_result, "inference_mode")
        assert inference_result.inference_mode is True

    def test_build_context_override_chaining(self) -> None:
        """Test BuildContext override chaining works correctly."""
        original_context = BuildContext(
            mode="training", overrides={"param1": "value1", "param2": "value2"}
        )

        # Chain overrides
        updated_context = original_context.with_overrides(param2="new_value2", param3="value3")
        final_context = updated_context.with_overrides(param1="final_value1")

        # Verify override chaining
        assert final_context.get_override("param1") == "final_value1"
        assert final_context.get_override("param2") == "new_value2"
        assert final_context.get_override("param3") == "value3"

        # Original should be unchanged
        assert original_context.get_override("param1") == "value1"
        assert original_context.get_override("param2") == "value2"
        assert original_context.get_override("param3") is None
