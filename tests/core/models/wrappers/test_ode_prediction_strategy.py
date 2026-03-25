"""Tests for ODEPredictionStrategy.

Covers:
- RuntimeError guard when configure_shape has not been called
- Output TensorDict keys: "predictions", "targets", "latents"
- Predictions shape (B, *data_shape)
- data_shape and n_steps properties
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict
from torch import Tensor, nn

from dlkit.core.models.nn.generative.functions.solvers import euler_step
from dlkit.core.models.nn.generative.samplers.noise import GaussianNoiseSampler
from dlkit.core.models.wrappers.prediction_strategies import ODEPredictionStrategy

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
_BATCH_SIZE: int = 4
_SPATIAL_DIM: int = 6
_N_STEPS: int = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinyVelocityModel(nn.Module):
    """Minimal velocity model: concatenates (x, t_expanded) → velocity."""

    def __init__(self, spatial_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(spatial_dim + 1, spatial_dim)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Compute velocity from (x, t).

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
def batch_size() -> int:
    """Batch size for ODE strategy tests.

    Returns:
        Integer batch size.
    """
    return _BATCH_SIZE


@pytest.fixture
def spatial_dim() -> int:
    """Spatial dimension for ODE strategy tests.

    Returns:
        Integer spatial dimension.
    """
    return _SPATIAL_DIM


@pytest.fixture
def ode_strategy(spatial_dim: int) -> ODEPredictionStrategy:
    """Configured ODEPredictionStrategy with euler solver.

    Args:
        spatial_dim: Spatial dimension fixture.

    Returns:
        Configured ODEPredictionStrategy.
    """
    strategy = ODEPredictionStrategy(
        x0_sampler=GaussianNoiseSampler(),
        solver=euler_step,
        n_steps=_N_STEPS,
    )
    strategy.configure_shape((spatial_dim,))
    return strategy


@pytest.fixture
def tiny_model(spatial_dim: int) -> _TinyVelocityModel:
    """Tiny velocity model fixture.

    Args:
        spatial_dim: Spatial dimension fixture.

    Returns:
        Instantiated _TinyVelocityModel.
    """
    return _TinyVelocityModel(spatial_dim)


@pytest.fixture
def minimal_batch(batch_size: int, spatial_dim: int) -> TensorDict:
    """Minimal TensorDict batch for predict calls.

    Args:
        batch_size: Batch size fixture.
        spatial_dim: Spatial dimension fixture.

    Returns:
        TensorDict with ``features["xt"]`` and empty ``targets``.
    """
    xt = torch.randn(batch_size, spatial_dim)
    return TensorDict(
        {
            "features": TensorDict({"xt": xt}, batch_size=[batch_size]),
            "targets": TensorDict({}, batch_size=[batch_size]),
        },
        batch_size=[batch_size],
    )


# ===========================================================================
# Guard: configure_shape must be called first
# ===========================================================================


def test_ode_strategy_raises_before_configure_shape(batch_size: int, spatial_dim: int) -> None:
    """ODEPredictionStrategy.predict raises RuntimeError if configure_shape not called.

    Args:
        batch_size: Batch size fixture.
        spatial_dim: Spatial dimension fixture.
    """
    strategy = ODEPredictionStrategy(
        x0_sampler=GaussianNoiseSampler(),
        solver=euler_step,
        n_steps=_N_STEPS,
    )
    model = _TinyVelocityModel(spatial_dim)
    batch = TensorDict(
        {
            "features": TensorDict(
                {"xt": torch.randn(batch_size, spatial_dim)}, batch_size=[batch_size]
            ),
            "targets": TensorDict({}, batch_size=[batch_size]),
        },
        batch_size=[batch_size],
    )
    with pytest.raises(RuntimeError, match="configure_shape"):
        strategy.predict(model, batch)


# ===========================================================================
# Good-path: output structure
# ===========================================================================


def test_ode_strategy_predict_output_keys(
    ode_strategy: ODEPredictionStrategy,
    tiny_model: _TinyVelocityModel,
    minimal_batch: TensorDict,
) -> None:
    """predict returns TensorDict with 'predictions', 'targets', 'latents' keys.

    Args:
        ode_strategy: Configured strategy fixture.
        tiny_model: Velocity model fixture.
        minimal_batch: Input batch fixture.
    """
    with torch.no_grad():
        result = ode_strategy.predict(tiny_model, minimal_batch)
    assert "predictions" in result.keys()
    assert "targets" in result.keys()
    assert "latents" in result.keys()


def test_ode_strategy_predictions_shape(
    ode_strategy: ODEPredictionStrategy,
    tiny_model: _TinyVelocityModel,
    minimal_batch: TensorDict,
    batch_size: int,
    spatial_dim: int,
) -> None:
    """predict returns predictions of shape (batch_size, spatial_dim).

    Args:
        ode_strategy: Configured strategy fixture.
        tiny_model: Velocity model fixture.
        minimal_batch: Input batch fixture.
        batch_size: Batch size fixture.
        spatial_dim: Spatial dimension fixture.
    """
    with torch.no_grad():
        result = ode_strategy.predict(tiny_model, minimal_batch)
    assert result["predictions"].shape == torch.Size([batch_size, spatial_dim])


def test_ode_strategy_latents_shape(
    ode_strategy: ODEPredictionStrategy,
    tiny_model: _TinyVelocityModel,
    minimal_batch: TensorDict,
    batch_size: int,
) -> None:
    """predict returns empty latents of shape (batch_size, 0).

    Args:
        ode_strategy: Configured strategy fixture.
        tiny_model: Velocity model fixture.
        minimal_batch: Input batch fixture.
        batch_size: Batch size fixture.
    """
    with torch.no_grad():
        result = ode_strategy.predict(tiny_model, minimal_batch)
    assert result["latents"].shape == torch.Size([batch_size, 0])


# ===========================================================================
# Properties
# ===========================================================================


def test_ode_strategy_data_shape_property(spatial_dim: int) -> None:
    """data_shape property returns the tuple set via configure_shape.

    Args:
        spatial_dim: Spatial dimension fixture.
    """
    strategy = ODEPredictionStrategy(
        x0_sampler=GaussianNoiseSampler(),
        solver=euler_step,
        n_steps=_N_STEPS,
    )
    assert strategy.data_shape is None
    strategy.configure_shape((spatial_dim,))
    assert strategy.data_shape == (spatial_dim,)


def test_ode_strategy_n_steps_property() -> None:
    """n_steps property returns the value supplied at construction."""
    n = 17
    strategy = ODEPredictionStrategy(
        x0_sampler=GaussianNoiseSampler(),
        solver=euler_step,
        n_steps=n,
    )
    assert strategy.n_steps == n


def test_ode_strategy_data_shape_none_before_configure() -> None:
    """data_shape is None before configure_shape is called."""
    strategy = ODEPredictionStrategy(
        x0_sampler=GaussianNoiseSampler(),
        solver=euler_step,
    )
    assert strategy.data_shape is None


def test_ode_strategy_reproducible_with_seeded_generator(
    ode_strategy: ODEPredictionStrategy,
    tiny_model: _TinyVelocityModel,
    minimal_batch: TensorDict,
) -> None:
    """Identical seeds produce identical predictions.

    Args:
        ode_strategy: Configured strategy fixture.
        tiny_model: Velocity model fixture.
        minimal_batch: Input batch fixture.
    """
    gen1 = torch.Generator()
    gen1.manual_seed(0)
    gen2 = torch.Generator()
    gen2.manual_seed(0)

    with torch.no_grad():
        r1 = ode_strategy.predict(tiny_model, minimal_batch, generator=gen1)
        r2 = ode_strategy.predict(tiny_model, minimal_batch, generator=gen2)

    assert torch.equal(r1["predictions"], r2["predictions"])
