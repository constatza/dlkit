"""Tests for RoutedMetricsUpdater device placement and routing correctness.

The critical invariant: metric state tensors must follow .to(device) from the
parent LightningModule. This is tested using the meta device as a portable
stand-in for any non-CPU device — no GPU required.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict
from torch import nn

from dlkit.domain.metrics.torchmetrics_wrappers import NormalizedVectorNormError
from dlkit.engine.adapters.lightning.metrics_routing import MetricRoute, RoutedMetricsUpdater

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def normalized_error_metric() -> NormalizedVectorNormError:
    """A NormalizedVectorNormError metric instance."""
    return NormalizedVectorNormError(vector_dim=-1, norm_ord=2)


@pytest.fixture
def val_route(normalized_error_metric: NormalizedVectorNormError) -> MetricRoute:
    """A single MetricRoute targeting targets['y'] for val stage."""
    return MetricRoute(
        metric=normalized_error_metric,
        target_ns="targets",
        target_name="y",
        extra_inputs=(),
    )


@pytest.fixture
def two_vector_batch() -> TensorDict:
    """Batch with targets['y'] of shape (2, 4)."""
    return TensorDict(
        {"targets": TensorDict({"y": torch.ones(2, 4)}, batch_size=[2])},
        batch_size=[2],
    )


@pytest.fixture
def two_vector_preds() -> torch.Tensor:
    """Predictions tensor with shape (2, 4)."""
    return torch.ones(2, 4)


# ---------------------------------------------------------------------------
# Device-placement test (CPU-reproducible root-cause regression)
# ---------------------------------------------------------------------------


def test_metric_state_device_follows_parent_to_call(
    val_route: MetricRoute,
) -> None:
    """Metric state tensors must move when the parent LightningModule moves devices.

    Uses meta device as a hardware-independent stand-in for any non-CPU target.
    Without the nn.Module registration fix, sum_errors and total stay on CPU
    while model outputs are on the target device, causing RuntimeError in update().
    """
    updater = RoutedMetricsUpdater(val_routes=[val_route], test_routes=[])

    class _Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._metrics_updater = updater

    _Wrapper().to("meta")

    metric = val_route.metric
    assert metric.sum_errors.device.type == "meta"
    assert metric.total.device.type == "meta"


# ---------------------------------------------------------------------------
# Routing correctness tests
# ---------------------------------------------------------------------------


def test_update_compute_roundtrip(
    val_route: MetricRoute,
    two_vector_preds: torch.Tensor,
    two_vector_batch: TensorDict,
) -> None:
    """update() followed by compute() returns a non-negative scalar."""
    updater = RoutedMetricsUpdater(val_routes=[val_route], test_routes=[])
    updater.update(two_vector_preds, two_vector_batch, "val")
    result = updater.compute("val")
    assert "NormalizedVectorNormError" in result
    assert float(result["NormalizedVectorNormError"]) >= 0.0


def test_reset_clears_accumulated_state(
    val_route: MetricRoute,
    two_vector_preds: torch.Tensor,
    two_vector_batch: TensorDict,
) -> None:
    """reset() must zero state so the next epoch starts from a clean slate."""
    updater = RoutedMetricsUpdater(val_routes=[val_route], test_routes=[])
    updater.update(two_vector_preds, two_vector_batch, "val")
    updater.reset("val")

    zero_batch = TensorDict(
        {"targets": TensorDict({"y": torch.zeros(2, 4)}, batch_size=[2])},
        batch_size=[2],
    )
    updater.update(torch.zeros(2, 4), zero_batch, "val")
    result = updater.compute("val")
    assert float(result["NormalizedVectorNormError"]) == pytest.approx(0.0, abs=1e-6)


def test_unknown_stage_is_silently_ignored(
    val_route: MetricRoute,
    two_vector_preds: torch.Tensor,
    two_vector_batch: TensorDict,
) -> None:
    """update/compute on an unconfigured stage must return empty without raising."""
    updater = RoutedMetricsUpdater(val_routes=[val_route], test_routes=[])
    updater.update(two_vector_preds, two_vector_batch, "train")
    assert updater.compute("train") == {}


def test_empty_routes_lifecycle() -> None:
    """RoutedMetricsUpdater with no routes must complete full update/compute/reset cycle."""
    updater = RoutedMetricsUpdater(val_routes=[], test_routes=[])
    batch = TensorDict({}, batch_size=[])
    updater.update(torch.zeros(2, 4), batch, "val")
    assert updater.compute("val") == {}
    updater.reset("val")
