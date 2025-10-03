"""Tests for components.model_components module.

This module tests the model component settings classes that provide
pure configuration without build methods, following SOLID principles.
"""

from __future__ import annotations

from typing import Any
import pytest
from hypothesis import given, strategies as st
from pydantic import ValidationError

from dlkit.tools.config.components.model_components import (
    ModelComponentSettings,
    MetricComponentSettings,
    LossComponentSettings,
    WrapperComponentSettings,
)


class TestMetricComponentSettings:
    """Test suite for MetricComponentSettings functionality."""

    def test_initialization_with_defaults(self) -> None:
        """Test MetricComponentSettings initialization with default values."""
        settings = MetricComponentSettings()

        assert settings.name == "MeanSquaredError"
        assert settings.module_path == "torchmetrics.regression"

    def test_initialization_with_custom_data(self, metric_component_data: dict[str, Any]) -> None:
        """Test MetricComponentSettings initialization with custom

        Args:
            metric_component_data: Metric component dataflow fixture
        """
        settings = MetricComponentSettings(**metric_component_data)

        assert settings.name == "Accuracy"
        assert settings.module_path == "torchmetrics.classification"
        assert settings.task == "multiclass"  # Extra field allowed
        assert settings.num_classes == 10  # Extra field allowed

    def test_get_init_kwargs_excludes_component_fields(
        self, metric_component_data: dict[str, Any]
    ) -> None:
        """Test get_init_kwargs excludes component-specific fields.

        Args:
            metric_component_data: Metric component dataflow fixture
        """
        settings = MetricComponentSettings(**metric_component_data)
        kwargs = settings.get_init_kwargs()

        assert "name" not in kwargs
        assert "module_path" not in kwargs
        assert "task" in kwargs
        assert "num_classes" in kwargs

    def test_alias_name_field_works(self) -> None:
        """Test name alias 'name' works correctly."""
        settings = MetricComponentSettings(name="CustomMetric")

        assert settings.name == "CustomMetric"


class TestLossComponentSettings:
    """Test suite for LossComponentSettings functionality."""

    def test_initialization_with_defaults(self) -> None:
        """Test LossComponentSettings initialization with default values."""
        settings = LossComponentSettings()

        assert settings.name == "mean_squared_error"
        assert settings.module_path == "torchmetrics.functional.regression"

    def test_initialization_with_custom_data(self, loss_component_data: dict[str, Any]) -> None:
        """Test LossComponentSettings initialization with custom

        Args:
            loss_component_data: Loss component dataflow fixture
        """
        settings = LossComponentSettings(**loss_component_data)

        assert settings.name == "CrossEntropyLoss"
        assert settings.module_path == "torch.nn"
        assert hasattr(settings, "weight")
        assert hasattr(settings, "ignore_index")

    def test_supports_callable_name(self) -> None:
        """Test LossComponentSettings supports callable component names."""

        def custom_loss(x, y):
            return (x - y).abs().mean()

        settings = LossComponentSettings(name=custom_loss)

        assert settings.name == custom_loss
        assert callable(settings.name)

    def test_get_init_kwargs_with_callable(self) -> None:
        """Test get_init_kwargs works with callable component names."""

        def custom_loss(x, y):
            return (x - y).abs().mean()

        settings = LossComponentSettings(name=custom_loss, custom_param="test_value")
        kwargs = settings.get_init_kwargs()

        assert "name" not in kwargs
        assert "module_path" not in kwargs
        assert "custom_param" in kwargs


class TestModelComponentSettings:
    """Test suite for ModelComponentSettings functionality."""

    def test_initialization_required_name(self) -> None:
        """Test ModelComponentSettings requires name."""
        with pytest.raises(ValidationError, match="name"):
            ModelComponentSettings()

    def test_initialization_with_basic_data(self, model_component_data: dict[str, Any]) -> None:
        """Test ModelComponentSettings initialization with basic

        Args:
            model_component_data: Model component dataflow fixture
        """
        settings = ModelComponentSettings(**model_component_data)

        assert settings.name == "TestModel"
        assert settings.module_path == "test.models"
        assert settings.heads == 8
        assert settings.num_layers == 6
        assert settings.latent_size == 256

    def test_initialization_with_checkpoint(
        self, model_component_with_checkpoint_data: dict[str, Any]
    ) -> None:
        """Test ModelComponentSettings initialization with checkpoint.

        Args:
            model_component_with_checkpoint_data: Model with checkpoint dataflow fixture
        """
        settings = ModelComponentSettings(**model_component_with_checkpoint_data)

        assert settings.checkpoint is not None and settings.checkpoint.name == "model.ckpt"

    def test_initialization_allows_extra_fields(self) -> None:
        """Test ModelComponentSettings allows extra fields (extra='allow')."""
        settings = ModelComponentSettings(
            name="ExtraModel",
            custom_param="custom_value",
            another_extra=123,
            nested_config={"key": "value"},
        )

        assert settings.name == "ExtraModel"
        assert hasattr(settings, "custom_param")
        assert hasattr(settings, "another_extra")
        assert hasattr(settings, "nested_config")

    def test_hyperparameter_fields_support(self, hyperparameter_model_data: dict[str, Any]) -> None:
        """Test ModelComponentSettings supports hyperparameter specifications.

        Args:
            hyperparameter_model_data: Model with hyperparameters dataflow fixture
        """
        settings = ModelComponentSettings(**hyperparameter_model_data)

        assert isinstance(settings.heads, dict)
        assert settings.heads["low"] == 4
        assert isinstance(settings.latent_size, dict)
        assert settings.latent_size["low"] == 64
        assert settings.latent_size["high"] == 512


    def test_supports_type_name(self) -> None:
        """Test ModelComponentSettings supports type objects as component names."""
        import torch.nn as nn

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x

        settings = ModelComponentSettings(name=TestModel)

        assert settings.name == TestModel

    def test_get_init_kwargs_excludes_model_specific_fields(
        self, model_component_data: dict[str, Any]
    ) -> None:
        """Test get_init_kwargs excludes model-specific configuration fields.

        Args:
            model_component_data: Model component dataflow fixture
        """
        settings = ModelComponentSettings(**model_component_data)
        kwargs = settings.get_init_kwargs()

        assert "name" not in kwargs
        assert "module_path" not in kwargs
        assert "heads" in kwargs
        assert "num_layers" in kwargs

    @given(st.integers(min_value=1, max_value=16), st.integers(min_value=1, max_value=10))
    def test_model_property_hyperparameter_values(self, heads: int, layers: int) -> None:
        """Property test: Model accepts valid hyperparameter values.

        Args:
            heads: Generated number of heads
            layers: Generated number of layers
        """
        settings = ModelComponentSettings(name="PropertyTestModel", heads=heads, num_layers=layers)

        assert settings.heads == heads
        assert settings.num_layers == layers


class TestWrapperComponentSettings:
    """Test suite for WrapperComponentSettings functionality."""

    def test_initialization_with_defaults(self) -> None:
        """Test WrapperComponentSettings initialization with default values."""
        settings = WrapperComponentSettings()

        assert settings.name == "StandardLightningWrapper"
        assert settings.module_path == "dlkit.core.models.wrappers"
        assert isinstance(settings.optimizer, object)  # Default factory
        assert isinstance(settings.scheduler, object)  # Default factory
        assert settings.train is True
        assert settings.test is True
        assert settings.predict is True

    def test_initialization_with_complete_data(
        self, wrapper_component_data: dict[str, Any]
    ) -> None:
        """Test WrapperComponentSettings initialization with complete

        Args:
            wrapper_component_data: Complete wrapper dataflow fixture
        """
        settings = WrapperComponentSettings(**wrapper_component_data)

        assert settings.name == "StandardWrapper"
        assert settings.optimizer.name == "Adam"
        assert settings.scheduler.name == "StepLR"
        assert settings.checkpoint is not None and settings.checkpoint.name == "wrapper.ckpt"
        assert settings.train is True
        assert settings.predict is False
        assert len(settings.metrics) == 2

    def test_initialization_with_hyperparameters(
        self, complex_wrapper_data: dict[str, Any]
    ) -> None:
        """Test WrapperComponentSettings handles nested hyperparameters.

        Args:
            complex_wrapper_data: Complex wrapper dataflow fixture
        """
        settings = WrapperComponentSettings(**complex_wrapper_data)

        assert settings.is_autoencoder is True
        assert settings.optimizer.lr == 0.001
        assert settings.optimizer.weight_decay == 0.01

    def test_has_checkpoint_property(self, wrapper_component_data: dict[str, Any]) -> None:
        """Test has_checkpoint property works correctly.

        Args:
            wrapper_component_data: Wrapper dataflow with checkpoint fixture
        """
        settings_with_checkpoint = WrapperComponentSettings(**wrapper_component_data)
        settings_without_checkpoint = WrapperComponentSettings()

        assert settings_with_checkpoint.has_checkpoint is True
        assert settings_without_checkpoint.has_checkpoint is False

    def test_has_metrics_property(self, wrapper_component_data: dict[str, Any]) -> None:
        """Test has_metrics property works correctly.

        Args:
            wrapper_component_data: Wrapper dataflow with metrics fixture
        """
        settings_with_metrics = WrapperComponentSettings(**wrapper_component_data)
        settings_without_metrics = WrapperComponentSettings()

        assert settings_with_metrics.has_metrics is True
        assert settings_without_metrics.has_metrics is False

    def test_nested_component_settings_validation(
        self, wrapper_component_data: dict[str, Any]
    ) -> None:
        """Test nested component settings are properly validated.

        Args:
            wrapper_component_data: Wrapper dataflow fixture
        """
        settings = WrapperComponentSettings(**wrapper_component_data)

        # Optimizer settings should be properly initialized
        assert settings.optimizer.name == "Adam"
        assert settings.optimizer.lr == 0.001

        # Loss function settings should be properly initialized
        assert settings.loss_function.name == "mse_loss"
        assert settings.loss_function.module_path == "torch.nn.functional"

        # Metrics should be properly initialized as tuple
        assert isinstance(settings.metrics, tuple)
        assert len(settings.metrics) == 2
        assert settings.metrics[0].name == "MeanSquaredError"

    def test_empty_collections_handled_correctly(self) -> None:
        """Test empty collections are handled correctly."""
        settings = WrapperComponentSettings(metrics=())

        assert len(settings.metrics) == 0
        assert settings.has_metrics is False

    def test_get_init_kwargs_excludes_wrapper_specific_fields(
        self, wrapper_component_data: dict[str, Any]
    ) -> None:
        """Test get_init_kwargs excludes wrapper-specific fields.

        Args:
            wrapper_component_data: Wrapper dataflow fixture
        """
        settings = WrapperComponentSettings(**wrapper_component_data)
        kwargs = settings.get_init_kwargs()

        assert "name" not in kwargs
        assert "module_path" not in kwargs
        assert "optimizer" in kwargs
        assert "scheduler" in kwargs
        assert "loss_function" in kwargs

    @given(st.booleans(), st.booleans(), st.booleans())
    def test_wrapper_property_training_flags(self, train: bool, test: bool, predict: bool) -> None:
        """Property test: Wrapper training flags work correctly.

        Args:
            train: Whether to enable training
            test: Whether to enable testing
            predict: Whether to enable prediction
        """
        settings = WrapperComponentSettings(train=train, test=test, predict=predict)

        assert settings.train == train
        assert settings.test == test
        assert settings.predict == predict
