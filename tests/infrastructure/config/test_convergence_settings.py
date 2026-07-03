"""Tests for ConvergenceSettings — good-path and validation.

Covers:
- resolved_sizes() with an explicit sizes list
- resolved_sizes() with log-space formula (min_samples / max_samples / steps)
- explicit list takes precedence when both are provided
- validator raises ValueError when neither branch is set
- repeats defaults to 1
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dlkit.infrastructure.config.convergence_settings import ConvergenceSettings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPLICIT_SIZES: tuple[int, ...] = (10, 50, 200, 1000)
LOG_MIN: int = 10
LOG_MAX: int = 1000
LOG_STEPS: int = 4


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def explicit_sizes_settings() -> ConvergenceSettings:
    """ConvergenceSettings with an explicit sizes tuple.

    Returns:
        ConvergenceSettings: Instance with sizes=(10, 50, 200, 1000).
    """
    return ConvergenceSettings(sizes=EXPLICIT_SIZES)


@pytest.fixture
def logspace_settings() -> ConvergenceSettings:
    """ConvergenceSettings using log-space formula with 4 steps.

    Returns:
        ConvergenceSettings: Instance with min_samples=10, max_samples=1000, steps=4.
    """
    return ConvergenceSettings(min_samples=LOG_MIN, max_samples=LOG_MAX, steps=LOG_STEPS)


@pytest.fixture
def both_branches_settings() -> ConvergenceSettings:
    """ConvergenceSettings with both explicit sizes and log-space fields set.

    Returns:
        ConvergenceSettings: Instance where explicit list should take precedence.
    """
    return ConvergenceSettings(
        sizes=EXPLICIT_SIZES,
        min_samples=LOG_MIN,
        max_samples=LOG_MAX,
        steps=LOG_STEPS,
    )


# ---------------------------------------------------------------------------
# Good-path tests
# ---------------------------------------------------------------------------


def test_resolved_sizes_explicit_returns_same_tuple(
    explicit_sizes_settings: ConvergenceSettings,
) -> None:
    """resolved_sizes() with explicit sizes returns the same tuple unchanged.

    Args:
        explicit_sizes_settings: Fixture with sizes=(10, 50, 200, 1000).
    """
    result = explicit_sizes_settings.resolved_sizes()
    assert result == EXPLICIT_SIZES


def test_resolved_sizes_logspace_returns_sorted_ints(
    logspace_settings: ConvergenceSettings,
) -> None:
    """resolved_sizes() with log-space formula returns a sorted tuple of ints.

    Args:
        logspace_settings: Fixture with min_samples=10, max_samples=1000, steps=4.
    """
    result = logspace_settings.resolved_sizes()

    assert len(result) >= 1
    assert all(isinstance(n, int) for n in result)
    assert result == tuple(sorted(result))
    assert result[0] >= LOG_MIN
    assert result[-1] <= LOG_MAX


def test_resolved_sizes_logspace_step_count(
    logspace_settings: ConvergenceSettings,
) -> None:
    """Log-space resolved_sizes() produces at most LOG_STEPS distinct values.

    Args:
        logspace_settings: Fixture with steps=4.
    """
    result = logspace_settings.resolved_sizes()
    assert len(result) <= LOG_STEPS


def test_explicit_list_takes_precedence_over_logspace(
    both_branches_settings: ConvergenceSettings,
) -> None:
    """When both sizes and log-space fields are provided, sizes wins.

    Args:
        both_branches_settings: Fixture with both branches set.
    """
    result = both_branches_settings.resolved_sizes()
    assert result == EXPLICIT_SIZES


def test_repeats_defaults_to_one() -> None:
    """ConvergenceSettings.repeats should default to 1."""
    settings = ConvergenceSettings(sizes=(100,))
    assert settings.repeats == 1


def test_repeats_can_be_overridden() -> None:
    """ConvergenceSettings.repeats accepts positive integers > 1."""
    settings = ConvergenceSettings(sizes=(100,), repeats=5)
    assert settings.repeats == 5


def test_target_defaults_to_none() -> None:
    """ConvergenceSettings.target should default to None (exploratory mode)."""
    settings = ConvergenceSettings(sizes=(100,))
    assert settings.target is None


# ---------------------------------------------------------------------------
# Negative / validation tests
# ---------------------------------------------------------------------------


def test_validator_raises_when_neither_branch_set() -> None:
    """Validator raises ValueError when neither sizes nor min/max are provided."""
    with pytest.raises((ValueError, ValidationError)):
        ConvergenceSettings()


def test_validator_raises_when_only_min_samples_set() -> None:
    """Validator raises when min_samples is set but max_samples is missing."""
    with pytest.raises((ValueError, ValidationError)):
        ConvergenceSettings(min_samples=10)


def test_validator_raises_when_only_max_samples_set() -> None:
    """Validator raises when max_samples is set but min_samples is missing."""
    with pytest.raises((ValueError, ValidationError)):
        ConvergenceSettings(max_samples=1000)
