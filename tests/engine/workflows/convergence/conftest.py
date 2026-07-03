"""Shared fixtures for convergence aggregation tests."""

from __future__ import annotations

import pytest

from dlkit.common.results import TrainingResult
from dlkit.infrastructure.config.convergence_settings import ConvergenceSettings

# ---------------------------------------------------------------------------
# Metric key constants
# ---------------------------------------------------------------------------

VAL_METRIC: str = "val/loss"
TRAIN_METRIC: str = "train/loss"

# ---------------------------------------------------------------------------
# Fixed metric values used across tests
# ---------------------------------------------------------------------------

VAL_LOSS_LOW: float = 0.03
VAL_LOSS_HIGH: float = 0.10
TRAIN_LOSS_LOW: float = 0.01
TRAIN_LOSS_HIGH: float = 0.07
DURATION: float = 1.0


# ---------------------------------------------------------------------------
# TrainingResult fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def single_result_low() -> TrainingResult:
    """Minimal TrainingResult with low val/loss (converging).

    Returns:
        TrainingResult: val/loss=0.03, train/loss=0.01.
    """
    return TrainingResult(
        model_state=None,
        metrics={VAL_METRIC: VAL_LOSS_LOW, TRAIN_METRIC: TRAIN_LOSS_LOW},
        artifacts={},
        duration_seconds=DURATION,
    )


@pytest.fixture
def single_result_high() -> TrainingResult:
    """Minimal TrainingResult with high val/loss (not converging).

    Returns:
        TrainingResult: val/loss=0.10, train/loss=0.07.
    """
    return TrainingResult(
        model_state=None,
        metrics={VAL_METRIC: VAL_LOSS_HIGH, TRAIN_METRIC: TRAIN_LOSS_HIGH},
        artifacts={},
        duration_seconds=DURATION,
    )


# ---------------------------------------------------------------------------
# ConvergenceSettings fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg_no_target() -> ConvergenceSettings:
    """ConvergenceSettings with no convergence target (pure exploratory mode).

    Returns:
        ConvergenceSettings: 1 repeat, no target.
    """
    return ConvergenceSettings(sizes=(100, 200), repeats=1)


@pytest.fixture
def cfg_with_target() -> ConvergenceSettings:
    """ConvergenceSettings with target=0.05 and c=2.0.

    Returns:
        ConvergenceSettings: 1 repeat, target=0.05.
    """
    return ConvergenceSettings(sizes=(100, 200), repeats=1, target=0.05, c=2.0)
