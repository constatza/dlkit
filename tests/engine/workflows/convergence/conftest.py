"""Shared fixtures for convergence aggregation and orchestrator tests."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from dlkit.common.results import TrainingResult
from dlkit.infrastructure.config.convergence_settings import ConvergenceSettings
from dlkit.infrastructure.config.job_config import ConvergenceJobConfig

FIXTURES = Path(__file__).parent.parent.parent.parent / "fixtures" / "jobs"

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


# ---------------------------------------------------------------------------
# ConvergenceJobConfig fixture (orchestrator tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def convergence_job_settings() -> ConvergenceJobConfig:
    """Real ConvergenceJobConfig loaded from the shared fixture TOML, trimmed to 2 sizes.

    Returns:
        ConvergenceJobConfig: sizes=(10, 20), repeats=1 — 2 total children,
        so ConvergenceOrchestrator._build_children() produces ids
        "n=10_r=0" and "n=20_r=0".
    """
    from dlkit.infrastructure.config.factories import load_job

    settings = load_job(FIXTURES / "convergence.toml")
    patched = settings.patch({"convergence": {"sizes": [10, 20], "repeats": 1}})
    return cast(ConvergenceJobConfig, patched)
