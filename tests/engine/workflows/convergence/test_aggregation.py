"""Tests for pure aggregation functions in convergence workflow.

Covers:
- aggregate_results() with 1 size, 1 repeat: single ConvergencePoint with correct gap
- aggregate_results() with 2 sizes, 1 repeat: second point has correct marginal_gain
- aggregate_results() with target set: converged=True when threshold is met
- aggregate_results() with no target: converged=None for all points
- find_n_star() with no converged points: returns None
- find_n_star() returns smallest n where converged=True
- build_summary_dict() includes n_star key and points list
"""

from __future__ import annotations

import pytest

from dlkit.common.metric_stages import MetricStage, metric_key
from dlkit.common.results import ConvergencePoint, TrainingResult
from dlkit.engine.workflows.convergence.aggregation import (
    aggregate_results,
    build_summary_dict,
    find_n_star,
)
from dlkit.infrastructure.config.convergence_settings import ConvergenceSettings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VAL_METRIC: str = "val/loss"
TRAIN_METRIC: str = "train/loss"

# Low-loss result — will converge when target=0.05, c=2.0, val_std=0 (single repeat)
VAL_LOSS_LOW: float = 0.03
TRAIN_LOSS_LOW: float = 0.01

# High-loss result — will NOT converge when target=0.05
VAL_LOSS_HIGH: float = 0.10
TRAIN_LOSS_HIGH: float = 0.07

DURATION: float = 1.0
SIZE_SMALL: int = 100
SIZE_LARGE: int = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(val_loss: float, train_loss: float) -> TrainingResult:
    """Build a minimal TrainingResult with given metric values.

    Args:
        val_loss: Value to record under 'val/loss'.
        train_loss: Value to record under 'train/loss'.

    Returns:
        TrainingResult: Frozen result with no model_state or artifacts.
    """
    return TrainingResult(
        model_state=None,
        metrics={VAL_METRIC: val_loss, TRAIN_METRIC: train_loss},
        artifacts={},
        duration_seconds=DURATION,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def result_low() -> TrainingResult:
    """TrainingResult with low val/loss=0.03 and train/loss=0.01.

    Returns:
        TrainingResult: Low-loss result suitable for convergence tests.
    """
    return _make_result(VAL_LOSS_LOW, TRAIN_LOSS_LOW)


@pytest.fixture
def result_high() -> TrainingResult:
    """TrainingResult with high val/loss=0.10 and train/loss=0.07.

    Returns:
        TrainingResult: High-loss result that will NOT meet the convergence target.
    """
    return _make_result(VAL_LOSS_HIGH, TRAIN_LOSS_HIGH)


@pytest.fixture
def cfg_no_target() -> ConvergenceSettings:
    """ConvergenceSettings with 1 repeat and no convergence target.

    Returns:
        ConvergenceSettings: Exploratory-mode settings.
    """
    return ConvergenceSettings(sizes=(SIZE_SMALL, SIZE_LARGE), repeats=1)


@pytest.fixture
def cfg_with_target() -> ConvergenceSettings:
    """ConvergenceSettings with target=0.05 and c=2.0, 1 repeat.

    Returns:
        ConvergenceSettings: Settings with a convergence threshold.
    """
    return ConvergenceSettings(sizes=(SIZE_SMALL, SIZE_LARGE), repeats=1, target=0.05, c=2.0)


@pytest.fixture
def single_point_no_target(
    result_low: TrainingResult,
    cfg_no_target: ConvergenceSettings,
) -> tuple[ConvergencePoint, ...]:
    """aggregate_results() output for 1 size / 1 repeat / no target.

    Args:
        result_low: Low-loss TrainingResult fixture.
        cfg_no_target: Settings with no convergence target.

    Returns:
        Tuple of a single ConvergencePoint.
    """
    return aggregate_results(
        results=(result_low,),
        sizes=(SIZE_SMALL,),
        cfg=ConvergenceSettings(sizes=(SIZE_SMALL,), repeats=1),
    )


@pytest.fixture
def two_point_no_target(
    result_high: TrainingResult,
    result_low: TrainingResult,
    cfg_no_target: ConvergenceSettings,
) -> tuple[ConvergencePoint, ...]:
    """aggregate_results() output for 2 sizes / 1 repeat / no target.

    Results order: size_small -> result_high, size_large -> result_low
    (val_mean decreases, so marginal_gain on second point should be positive).

    Args:
        result_high: High-loss result for the small size.
        result_low: Low-loss result for the large size.
        cfg_no_target: Settings with no convergence target.

    Returns:
        Tuple of two ConvergencePoints.
    """
    return aggregate_results(
        results=(result_high, result_low),
        sizes=(SIZE_SMALL, SIZE_LARGE),
        cfg=cfg_no_target,
    )


@pytest.fixture
def two_point_with_target_mixed(
    result_high: TrainingResult,
    result_low: TrainingResult,
    cfg_with_target: ConvergenceSettings,
) -> tuple[ConvergencePoint, ...]:
    """aggregate_results() with target: first NOT converged, second converged.

    With target=0.05, c=2.0, val_std=0 (single repeat):
    - result_high: val_mean=0.10 -> 0.10 + 0 > 0.05 -> NOT converged
    - result_low:  val_mean=0.03 -> 0.03 + 0 <= 0.05 -> converged

    Args:
        result_high: High val/loss result.
        result_low: Low val/loss result.
        cfg_with_target: Settings with convergence target.

    Returns:
        Tuple of two ConvergencePoints where the second has converged=True.
    """
    return aggregate_results(
        results=(result_high, result_low),
        sizes=(SIZE_SMALL, SIZE_LARGE),
        cfg=cfg_with_target,
    )


# ---------------------------------------------------------------------------
# Tests: aggregate_results()
# ---------------------------------------------------------------------------


def test_aggregate_single_size_returns_one_point(
    single_point_no_target: tuple[ConvergencePoint, ...],
) -> None:
    """aggregate_results() with 1 size returns exactly one ConvergencePoint.

    Args:
        single_point_no_target: Fixture with single point.
    """
    assert len(single_point_no_target) == 1


def test_aggregate_single_size_correct_n(
    single_point_no_target: tuple[ConvergencePoint, ...],
) -> None:
    """The ConvergencePoint n field matches the size used.

    Args:
        single_point_no_target: Fixture with single point.
    """
    assert single_point_no_target[0].n == SIZE_SMALL


def test_aggregate_single_size_correct_gap(
    single_point_no_target: tuple[ConvergencePoint, ...],
) -> None:
    """gap = val_mean - train_mean for a single-repeat result.

    Args:
        single_point_no_target: Fixture with single point.
    """
    point = single_point_no_target[0]
    expected_gap = VAL_LOSS_LOW - TRAIN_LOSS_LOW
    assert abs(point.gap - expected_gap) < 1e-9


def test_aggregate_single_size_first_marginal_gain_is_none(
    single_point_no_target: tuple[ConvergencePoint, ...],
) -> None:
    """First point always has marginal_gain=None (no previous step to compare).

    Args:
        single_point_no_target: Fixture with single point.
    """
    assert single_point_no_target[0].marginal_gain is None


def test_aggregate_two_sizes_second_has_marginal_gain(
    two_point_no_target: tuple[ConvergencePoint, ...],
) -> None:
    """Second ConvergencePoint has a non-None marginal_gain.

    Args:
        two_point_no_target: Fixture with two points.
    """
    assert two_point_no_target[1].marginal_gain is not None


def test_aggregate_two_sizes_marginal_gain_value(
    two_point_no_target: tuple[ConvergencePoint, ...],
) -> None:
    """marginal_gain == prev_val_mean - current_val_mean (improvement = positive).

    Args:
        two_point_no_target: Fixture with two points (high->low val_loss).
    """
    first, second = two_point_no_target
    expected = first.val_mean - second.val_mean
    assert abs(second.marginal_gain - expected) < 1e-9


def test_aggregate_no_target_converged_is_none(
    two_point_no_target: tuple[ConvergencePoint, ...],
) -> None:
    """All points have converged=None when no target is configured.

    Args:
        two_point_no_target: Fixture with no target set.
    """
    assert all(p.converged is None for p in two_point_no_target)


def test_aggregate_with_target_first_not_converged(
    two_point_with_target_mixed: tuple[ConvergencePoint, ...],
) -> None:
    """First point is not converged when val_mean exceeds threshold.

    Args:
        two_point_with_target_mixed: Mixed fixture, first point high-loss.
    """
    assert two_point_with_target_mixed[0].converged is False


def test_aggregate_with_target_second_converged(
    two_point_with_target_mixed: tuple[ConvergencePoint, ...],
) -> None:
    """Second point converged=True when val_mean + c*val_std <= target.

    With single repeat: val_std=0, so val_mean=0.03 <= target=0.05.

    Args:
        two_point_with_target_mixed: Mixed fixture, second point low-loss.
    """
    assert two_point_with_target_mixed[1].converged is True


def test_aggregate_reads_real_default_target_metric() -> None:
    """aggregate_results() honours ConvergenceSettings' real default target_metric.

    Regression test: previously the train-metric counterpart was derived via
    ``cfg.target_metric.replace("val/", "train/")``, an inline literal that
    assumed a "/" separator never actually guaranteed anywhere. Metric keys
    here are built with the shared `metric_key()` helper (the same convention
    runtime logging uses via `_format_metric_name`), not the local
    VAL_METRIC/TRAIN_METRIC constants, so this test fails loudly if the
    lookup and the `target_metric` default ever drift apart -- instead of
    aggregate_results() silently falling back to a 0.0 mean for every value.
    """
    val_key = metric_key(MetricStage.VAL, "loss")
    train_key = metric_key(MetricStage.TRAIN, "loss")
    result = TrainingResult(
        model_state=None,
        metrics={val_key: VAL_LOSS_LOW, train_key: TRAIN_LOSS_LOW},
        artifacts={},
        duration_seconds=DURATION,
    )
    cfg = ConvergenceSettings(sizes=(SIZE_SMALL,), repeats=1)

    points = aggregate_results(results=(result,), sizes=(SIZE_SMALL,), cfg=cfg)

    assert points[0].val_mean == pytest.approx(VAL_LOSS_LOW)
    assert points[0].train_mean == pytest.approx(TRAIN_LOSS_LOW)


# ---------------------------------------------------------------------------
# Tests: find_n_star()
# ---------------------------------------------------------------------------


def test_find_n_star_no_converged_returns_none(
    two_point_no_target: tuple[ConvergencePoint, ...],
) -> None:
    """find_n_star() returns None when no point has converged=True.

    Args:
        two_point_no_target: Points where converged=None everywhere.
    """
    result = find_n_star(two_point_no_target)
    assert result is None


def test_find_n_star_all_false_returns_none() -> None:
    """find_n_star() returns None when all points have converged=False."""
    points = (
        ConvergencePoint(
            n=100,
            train_mean=0.07,
            train_std=0.0,
            val_mean=0.10,
            val_std=0.0,
            gap=0.03,
            marginal_gain=None,
            converged=False,
        ),
        ConvergencePoint(
            n=200,
            train_mean=0.06,
            train_std=0.0,
            val_mean=0.08,
            val_std=0.0,
            gap=0.02,
            marginal_gain=0.02,
            converged=False,
        ),
    )
    assert find_n_star(points) is None


def test_find_n_star_returns_first_converged_n(
    two_point_with_target_mixed: tuple[ConvergencePoint, ...],
) -> None:
    """find_n_star() returns the n of the first converged point.

    Args:
        two_point_with_target_mixed: Mixed fixture; second point converges at SIZE_LARGE.
    """
    n_star = find_n_star(two_point_with_target_mixed)
    assert n_star == SIZE_LARGE


def test_find_n_star_returns_smallest_n_when_multiple_converge() -> None:
    """find_n_star() returns the smallest n (first in tuple) when multiple points converge."""
    points = (
        ConvergencePoint(
            n=50,
            train_mean=0.01,
            train_std=0.0,
            val_mean=0.02,
            val_std=0.0,
            gap=0.01,
            marginal_gain=None,
            converged=True,
        ),
        ConvergencePoint(
            n=100,
            train_mean=0.01,
            train_std=0.0,
            val_mean=0.02,
            val_std=0.0,
            gap=0.01,
            marginal_gain=0.0,
            converged=True,
        ),
    )
    assert find_n_star(points) == 50


# ---------------------------------------------------------------------------
# Tests: build_summary_dict()
# ---------------------------------------------------------------------------


def test_build_summary_dict_has_n_star_key(
    two_point_with_target_mixed: tuple[ConvergencePoint, ...],
) -> None:
    """build_summary_dict() output contains the 'n_star' key.

    Args:
        two_point_with_target_mixed: Mixed fixture.
    """
    n_star = find_n_star(two_point_with_target_mixed)
    summary = build_summary_dict(two_point_with_target_mixed, n_star)
    assert "n_star" in summary


def test_build_summary_dict_has_points_key(
    two_point_with_target_mixed: tuple[ConvergencePoint, ...],
) -> None:
    """build_summary_dict() output contains the 'points' key.

    Args:
        two_point_with_target_mixed: Mixed fixture.
    """
    n_star = find_n_star(two_point_with_target_mixed)
    summary = build_summary_dict(two_point_with_target_mixed, n_star)
    assert "points" in summary


def test_build_summary_dict_n_star_value(
    two_point_with_target_mixed: tuple[ConvergencePoint, ...],
) -> None:
    """build_summary_dict() n_star matches find_n_star() output.

    Args:
        two_point_with_target_mixed: Mixed fixture; n_star should be SIZE_LARGE.
    """
    n_star = find_n_star(two_point_with_target_mixed)
    summary = build_summary_dict(two_point_with_target_mixed, n_star)
    assert summary["n_star"] == SIZE_LARGE


def test_build_summary_dict_points_count(
    two_point_with_target_mixed: tuple[ConvergencePoint, ...],
) -> None:
    """build_summary_dict() points list has one entry per ConvergencePoint.

    Args:
        two_point_with_target_mixed: Fixture with 2 points.
    """
    n_star = find_n_star(two_point_with_target_mixed)
    summary = build_summary_dict(two_point_with_target_mixed, n_star)
    assert len(summary["points"]) == 2


def test_build_summary_dict_none_n_star() -> None:
    """build_summary_dict() with n_star=None stores None under 'n_star' key."""
    points: tuple[ConvergencePoint, ...] = ()
    summary = build_summary_dict(points, None)
    assert summary["n_star"] is None
