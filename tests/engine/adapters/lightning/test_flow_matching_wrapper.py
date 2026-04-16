"""Integration tests for FlowMatchingWrapper.

Covers:
- training_step returns a dict with a scalar "loss" tensor
- validation_step runs without error and returns "val_loss"
- predict_step returns a TensorDict with "predictions", "targets", "latents"
- predict_step predictions shape is (batch_size, spatial_dim)

The wrapper is assembled from first-class protocol objects without using the
build factory, keeping the test focused and independent of configuration loading.
"""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor, nn

from dlkit.domain.nn.generative.functions.solvers import euler_step
from dlkit.domain.nn.generative.samplers.noise import GaussianNoiseSampler
from dlkit.domain.nn.generative.supervision import FlowMatchingSupervisionBuilder
from dlkit.engine.adapters.lightning.flowmatching import FlowMatchingWrapper
from dlkit.engine.adapters.lightning.generator_factories import DeterministicGeneratorFactory
from dlkit.engine.adapters.lightning.loss_routing import RoutedLossComputer
from dlkit.engine.adapters.lightning.metrics_routing import RoutedMetricsUpdater
from dlkit.engine.adapters.lightning.model_invoker import ModelOutputSpec, TensorDictModelInvoker
from dlkit.engine.adapters.lightning.prediction_strategies import ODEPredictionStrategy
from dlkit.engine.adapters.lightning.transform_pipeline import NamedBatchTransformer
from dlkit.engine.adapters.lightning.wrapper_types import WrapperCheckpointMetadata
from dlkit.infrastructure.config.model_components import WrapperComponentSettings
from dlkit.infrastructure.config.optimizer_settings import OptimizerSettings

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
_SPATIAL_DIM: int = 4
_BATCH_SIZE: int = 2
_N_ODE_STEPS: int = 5
_BASE_SEED: int = 42


# ---------------------------------------------------------------------------
# Tiny velocity model
# ---------------------------------------------------------------------------


class _TinyVelocityModel(nn.Module):
    """Velocity model that accepts (xt, t) positionally and returns a velocity.

    Concatenates the state tensor and time along the last dimension then
    maps to a velocity via a single linear layer.
    """

    def __init__(self, spatial_dim: int) -> None:
        """Initialise the linear velocity network.

        Args:
            spatial_dim: Spatial feature dimension of the data.
        """
        super().__init__()
        self.fc = nn.Linear(spatial_dim + 1, spatial_dim)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Compute velocity from state and time.

        Args:
            x: State tensor of shape ``(B, spatial_dim)``.
            t: Time tensor of shape ``(B,)``.

        Returns:
            Velocity tensor of shape ``(B, spatial_dim)``.
        """
        t_exp = t.unsqueeze(-1).to(dtype=x.dtype)
        return self.fc(torch.cat([x, t_exp], dim=-1))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def spatial_dim() -> int:
    """Spatial dimension for wrapper tests.

    Returns:
        Integer spatial dimension.
    """
    return _SPATIAL_DIM


@pytest.fixture
def batch_size() -> int:
    """Batch size for wrapper tests.

    Returns:
        Integer batch size.
    """
    return _BATCH_SIZE


@pytest.fixture
def velocity_model(spatial_dim: int) -> _TinyVelocityModel:
    """Tiny velocity model fixture.

    Args:
        spatial_dim: Spatial dimension fixture.

    Returns:
        _TinyVelocityModel with float32 weights.
    """
    torch.manual_seed(_BASE_SEED)
    return _TinyVelocityModel(spatial_dim)


@pytest.fixture
def output_spec() -> ModelOutputSpec:
    """Default single-output ModelOutputSpec.

    Returns:
        ModelOutputSpec with prediction_key="predictions".
    """
    return ModelOutputSpec()


@pytest.fixture
def model_invoker(output_spec: ModelOutputSpec) -> TensorDictModelInvoker:
    """TensorDictModelInvoker reading (features.xt, features.t) positionally.

    Args:
        output_spec: Output spec fixture.

    Returns:
        Configured TensorDictModelInvoker.
    """
    return TensorDictModelInvoker(
        in_keys=[("features", "xt"), ("features", "t")],
        output_spec=output_spec,
    )


@pytest.fixture
def supervision_builder() -> FlowMatchingSupervisionBuilder:
    """FlowMatchingSupervisionBuilder reading x1 from features.

    Returns:
        FlowMatchingSupervisionBuilder for key "x1".
    """
    return FlowMatchingSupervisionBuilder(x1_key="x1")


@pytest.fixture
def ode_strategy(spatial_dim: int) -> ODEPredictionStrategy:
    """Configured ODEPredictionStrategy.

    Args:
        spatial_dim: Spatial dimension fixture.

    Returns:
        ODEPredictionStrategy with euler solver, pre-configured shape.
    """
    strategy = ODEPredictionStrategy(
        x0_sampler=GaussianNoiseSampler(),
        solver=euler_step,
        n_steps=_N_ODE_STEPS,
    )
    strategy.configure_shape((spatial_dim,))
    return strategy


@pytest.fixture
def checkpoint_metadata(
    output_spec: ModelOutputSpec,
) -> WrapperCheckpointMetadata:
    """Minimal WrapperCheckpointMetadata for the flow matching wrapper.

    Args:
        output_spec: Output spec fixture.

    Returns:
        WrapperCheckpointMetadata with mock model settings.
    """
    model_settings = MagicMock()
    wrapper_settings = WrapperComponentSettings(
        optimizer=OptimizerSettings(name="Adam"),
    )
    return WrapperCheckpointMetadata(
        model_settings=model_settings,
        wrapper_settings=wrapper_settings,
        entry_configs=(),
        feature_names=("x1",),
        predict_target_key="ut",
        shape_summary=None,
        output_spec=output_spec,
    )


@pytest.fixture
def flow_wrapper(
    velocity_model: _TinyVelocityModel,
    model_invoker: TensorDictModelInvoker,
    ode_strategy: ODEPredictionStrategy,
    supervision_builder: FlowMatchingSupervisionBuilder,
    checkpoint_metadata: WrapperCheckpointMetadata,
) -> FlowMatchingWrapper:
    """Fully assembled FlowMatchingWrapper ready for step calls.

    All components are constructed from fixtures; no factory or config loading.

    Args:
        velocity_model: Model fixture.
        model_invoker: Invoker fixture.
        ode_strategy: ODE prediction strategy fixture.
        supervision_builder: Supervision builder fixture.
        checkpoint_metadata: Checkpoint metadata fixture.

    Returns:
        Instantiated FlowMatchingWrapper.
    """
    loss_computer = RoutedLossComputer(
        loss_fn=F.mse_loss,
        target_key=None,
        default_target_key="ut",
        extra_inputs=(),
    )
    metrics_updater = RoutedMetricsUpdater(val_routes=[], test_routes=[])
    batch_transformer = NamedBatchTransformer({}, {})
    optimizer_settings = OptimizerSettings(name="Adam")

    return FlowMatchingWrapper(
        model=velocity_model,
        model_invoker=model_invoker,
        loss_computer=loss_computer,
        metrics_updater=metrics_updater,
        batch_transformer=batch_transformer,
        optimizer_settings=optimizer_settings,
        scheduler_settings=None,
        predict_target_key="ut",
        checkpoint_metadata=checkpoint_metadata,
        ode_prediction_strategy=ode_strategy,
        supervision_builder=supervision_builder,
        val_generator_factory=DeterministicGeneratorFactory(base_seed=_BASE_SEED),
    )


@pytest.fixture
def raw_x1_batch(batch_size: int, spatial_dim: int) -> TensorDict:
    """Input batch containing only features["x1"] (pre-supervision-builder).

    Args:
        batch_size: Batch size fixture.
        spatial_dim: Spatial dimension fixture.

    Returns:
        TensorDict with ``features["x1"]`` and empty ``targets``.
    """
    torch.manual_seed(_BASE_SEED)
    x1 = torch.randn(batch_size, spatial_dim)
    return TensorDict(
        {
            "features": TensorDict({"x1": x1}, batch_size=[batch_size]),
            "targets": TensorDict({}, batch_size=[batch_size]),
        },
        batch_size=[batch_size],
    )


# ===========================================================================
# training_step
# ===========================================================================


def test_training_step_returns_scalar_loss(
    flow_wrapper: FlowMatchingWrapper, raw_x1_batch: TensorDict
) -> None:
    """training_step returns a dict with a scalar Tensor under key 'loss'.

    Args:
        flow_wrapper: Assembled wrapper fixture.
        raw_x1_batch: Input batch fixture.
    """
    result = flow_wrapper.training_step(raw_x1_batch, 0)
    assert "loss" in result
    loss = result["loss"]
    assert isinstance(loss, Tensor)
    assert loss.shape == torch.Size([])


def test_training_step_loss_is_finite(
    flow_wrapper: FlowMatchingWrapper, raw_x1_batch: TensorDict
) -> None:
    """training_step loss is a finite (non-NaN, non-Inf) scalar.

    Args:
        flow_wrapper: Assembled wrapper fixture.
        raw_x1_batch: Input batch fixture.
    """
    result = flow_wrapper.training_step(raw_x1_batch, 0)
    assert torch.isfinite(result["loss"])


# ===========================================================================
# validation_step
# ===========================================================================


def test_validation_step_does_not_raise(
    flow_wrapper: FlowMatchingWrapper, raw_x1_batch: TensorDict
) -> None:
    """validation_step completes without raising any exception.

    Args:
        flow_wrapper: Assembled wrapper fixture.
        raw_x1_batch: Input batch fixture.
    """
    result = flow_wrapper.validation_step(raw_x1_batch, 0)
    assert "val_loss" in result


def test_validation_step_loss_is_finite(
    flow_wrapper: FlowMatchingWrapper, raw_x1_batch: TensorDict
) -> None:
    """validation_step val_loss is a finite scalar.

    Args:
        flow_wrapper: Assembled wrapper fixture.
        raw_x1_batch: Input batch fixture.
    """
    result = flow_wrapper.validation_step(raw_x1_batch, 0)
    assert torch.isfinite(result["val_loss"])


# ===========================================================================
# predict_step
# ===========================================================================


def test_predict_step_output_keys(
    flow_wrapper: FlowMatchingWrapper, raw_x1_batch: TensorDict
) -> None:
    """predict_step returns TensorDict with 'predictions', 'targets', 'latents'.

    Args:
        flow_wrapper: Assembled wrapper fixture.
        raw_x1_batch: Input batch fixture.
    """
    with torch.no_grad():
        result = flow_wrapper.predict_step(raw_x1_batch, 0)
    assert "predictions" in result.keys()
    assert "targets" in result.keys()
    assert "latents" in result.keys()


def test_predict_step_predictions_shape(
    flow_wrapper: FlowMatchingWrapper,
    raw_x1_batch: TensorDict,
    batch_size: int,
    spatial_dim: int,
) -> None:
    """predict_step predictions shape is (batch_size, spatial_dim).

    Args:
        flow_wrapper: Assembled wrapper fixture.
        raw_x1_batch: Input batch fixture.
        batch_size: Batch size fixture.
        spatial_dim: Spatial dimension fixture.
    """
    with torch.no_grad():
        result = flow_wrapper.predict_step(raw_x1_batch, 0)
    assert result["predictions"].shape == torch.Size([batch_size, spatial_dim])


def test_predict_step_predictions_finite(
    flow_wrapper: FlowMatchingWrapper, raw_x1_batch: TensorDict
) -> None:
    """predict_step predictions contain no NaN or Inf values.

    Args:
        flow_wrapper: Assembled wrapper fixture.
        raw_x1_batch: Input batch fixture.
    """
    with torch.no_grad():
        result = flow_wrapper.predict_step(raw_x1_batch, 0)
    assert torch.isfinite(cast("Tensor", result["predictions"])).all()
