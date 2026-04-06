"""Integration test for custom metrics with MetricCollection and settings.

Tests that our new torchmetrics-compatible custom metrics work end-to-end
with the config system and MetricCollection.
"""

from typing import Any, cast

import torch
from torchmetrics import MetricCollection

from dlkit.domain.metrics import (
    MeanSquaredError,
    NormalizedVectorNormError,
    TemporalDerivativeError,
)
from dlkit.infrastructure.config import BuildContext
from dlkit.infrastructure.config.core.factories import FactoryProvider
from dlkit.infrastructure.config.model_components import MetricComponentSettings


class TestCustomMetricsIntegration:
    """Integration tests for custom metrics with config system."""

    def test_metric_collection_with_custom_metrics(self):
        """Test MetricCollection works with our custom metrics."""
        # Create MetricCollection with mix of standard and custom metrics
        metrics = MetricCollection(
            {
                "mse": MeanSquaredError(),
                "norm_l2": NormalizedVectorNormError(norm_ord=2),
            }
        )

        # Test data
        preds = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.1, 1.9], [3.1, 3.9]])

        # Update and compute
        metrics.update(preds, target)
        results = metrics.compute()

        # Verify both metrics returned values
        assert "mse" in results
        assert "norm_l2" in results
        assert results["mse"].item() > 0
        assert results["norm_l2"].item() > 0

    def test_temporal_metric_in_collection(self):
        """Test temporal derivative metric in MetricCollection."""
        metrics = MetricCollection(
            {
                "velocity_error": TemporalDerivativeError(n=1, derivative_dim=1),
                "accel_error": TemporalDerivativeError(n=2, derivative_dim=1),
            }
        )

        # 3D temporal data (B=2, T=5, D=2)
        preds = torch.randn(2, 5, 2)
        target = torch.randn(2, 5, 2)

        metrics.update(preds, target)
        results = metrics.compute()

        assert "velocity_error" in results
        assert "accel_error" in results
        assert results["velocity_error"].item() >= 0
        assert results["accel_error"].item() >= 0

    def test_metric_created_via_factory(self):
        """Test metrics can be created via FactoryProvider (config system)."""
        # Create metric settings
        metric_settings = MetricComponentSettings.model_validate(
            {
                "name": "NormalizedVectorNormError",
                "module_path": "dlkit.domain.metrics",
                "norm_ord": 2,
            }
        )

        # Create via factory
        context = BuildContext(mode="training")
        metric = FactoryProvider.create_component(metric_settings, context)

        # Verify it's a torchmetrics.Metric instance
        assert hasattr(metric, "update")
        assert hasattr(metric, "compute")
        assert hasattr(metric, "reset")

        # Test it works
        preds = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.1, 1.9], [3.1, 3.9]])

        metric.update(preds, target)
        result = metric.compute()

        assert result.item() > 0

    def test_multiple_metrics_via_factory(self):
        """Test multiple metrics created via factory work in MetricCollection."""
        # Create metric settings
        mse_settings = MetricComponentSettings(
            name="MeanSquaredError",
            module_path="dlkit.domain.metrics",
        )
        norm_settings = MetricComponentSettings.model_validate(
            {
                "name": "NormalizedVectorNormError",
                "module_path": "dlkit.domain.metrics",
                "norm_ord": 2,
            }
        )

        # Create via factory
        context = BuildContext(mode="training")
        mse = FactoryProvider.create_component(mse_settings, context)
        norm_metric = FactoryProvider.create_component(norm_settings, context)

        # Put in MetricCollection
        metrics = MetricCollection(
            {
                "mse": mse,
                "norm_l2": norm_metric,
            }
        )

        # Test data
        preds = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.1, 1.9], [3.1, 3.9]])

        metrics.update(preds, target)
        results = metrics.compute()

        assert "mse" in results
        assert "norm_l2" in results
        assert results["mse"].item() > 0
        assert results["norm_l2"].item() > 0

    def test_metric_reset_functionality(self):
        """Test metric reset works correctly."""
        metric = NormalizedVectorNormError(norm_ord=2)

        # First batch
        preds1 = torch.tensor([[1.0, 2.0]])
        target1 = torch.tensor([[1.1, 1.9]])
        cast(Any, metric).update(preds1, target1)
        result1 = cast(Any, metric).compute()

        # Reset
        metric.reset()

        # Second batch (same data, should give same result)
        cast(Any, metric).update(preds1, target1)
        result2 = cast(Any, metric).compute()

        assert torch.allclose(result1, result2)

    def test_batch_accumulation(self):
        """Test metric accumulates correctly across batches."""
        metric = NormalizedVectorNormError(norm_ord=2)

        # Two small batches
        preds1 = torch.tensor([[1.0, 2.0]])
        target1 = torch.tensor([[1.1, 1.9]])
        preds2 = torch.tensor([[3.0, 4.0]])
        target2 = torch.tensor([[3.1, 3.9]])

        # Accumulate
        cast(Any, metric).update(preds1, target1)
        cast(Any, metric).update(preds2, target2)
        multi_batch_result = cast(Any, metric).compute()

        # Single large batch
        metric.reset()
        preds_all = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target_all = torch.tensor([[1.1, 1.9], [3.1, 3.9]])
        cast(Any, metric).update(preds_all, target_all)
        single_batch_result = cast(Any, metric).compute()

        # Should be equal (accumulation is correct)
        assert torch.allclose(multi_batch_result, single_batch_result)
