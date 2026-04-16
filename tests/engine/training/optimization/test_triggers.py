"""Tests for optimization stage transition triggers."""

from __future__ import annotations

from typing import Literal, cast

import pytest

from dlkit.engine.training.optimization.triggers import (
    EpochTransitionTrigger,
    NoTransitionTrigger,
    PlateauTransitionTrigger,
)

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def epoch_trigger_at_5() -> EpochTransitionTrigger:
    """Epoch trigger that fires at epoch 5.

    Returns:
        An EpochTransitionTrigger configured to fire at epoch 5.
    """
    return EpochTransitionTrigger(at_epoch=5)


@pytest.fixture
def epoch_trigger_at_0() -> EpochTransitionTrigger:
    """Epoch trigger that fires at epoch 0.

    Returns:
        An EpochTransitionTrigger configured to fire at epoch 0.
    """
    return EpochTransitionTrigger(at_epoch=0)


@pytest.fixture
def plateau_trigger_patience_2() -> PlateauTransitionTrigger:
    """Plateau trigger with patience=2 monitoring "val_loss" in min mode.

    Returns:
        A PlateauTransitionTrigger with patience=2, monitoring "val_loss".
    """
    return PlateauTransitionTrigger(
        monitor="val_loss",
        patience=2,
        min_delta=1e-4,
        mode="min",
    )


@pytest.fixture
def plateau_trigger_patience_1() -> PlateauTransitionTrigger:
    """Plateau trigger with patience=1 monitoring "val_loss" in min mode.

    Returns:
        A PlateauTransitionTrigger with patience=1, monitoring "val_loss".
    """
    return PlateauTransitionTrigger(
        monitor="val_loss",
        patience=1,
        min_delta=1e-4,
        mode="min",
    )


@pytest.fixture
def plateau_trigger_max_mode() -> PlateauTransitionTrigger:
    """Plateau trigger in max mode monitoring "val_acc".

    Returns:
        A PlateauTransitionTrigger configured to maximize "val_acc".
    """
    return PlateauTransitionTrigger(
        monitor="val_acc",
        patience=2,
        min_delta=1e-4,
        mode="max",
    )


@pytest.fixture
def no_trigger() -> NoTransitionTrigger:
    """No-op transition trigger.

    Returns:
        A NoTransitionTrigger instance.
    """
    return NoTransitionTrigger()


# ===========================================================================
# EpochTransitionTrigger tests
# ===========================================================================


def test_epoch_trigger_fires_at_target_epoch(epoch_trigger_at_5: EpochTransitionTrigger) -> None:
    """Test that epoch trigger fires when epoch reaches target.

    Args:
        epoch_trigger_at_5: Trigger configured for epoch 5.
    """
    # Act & Assert - Should not fire before target
    assert epoch_trigger_at_5.update(epoch=4, metrics={}) is False

    # Act & Assert - Should fire at target epoch
    assert epoch_trigger_at_5.update(epoch=5, metrics={}) is True


def test_epoch_trigger_does_not_fire_before_target(
    epoch_trigger_at_5: EpochTransitionTrigger,
) -> None:
    """Test that epoch trigger does not fire before target epoch.

    Args:
        epoch_trigger_at_5: Trigger configured for epoch 5.
    """
    # Act & Assert - Multiple updates below target should not fire
    for epoch in range(5):
        assert epoch_trigger_at_5.update(epoch=epoch, metrics={}) is False


def test_epoch_trigger_fires_only_once_without_reset(
    epoch_trigger_at_5: EpochTransitionTrigger,
) -> None:
    """Test that epoch trigger fires exactly once without reset.

    Args:
        epoch_trigger_at_5: Trigger configured for epoch 5.
    """
    # Act - Fire once at epoch 5
    assert epoch_trigger_at_5.update(epoch=5, metrics={}) is True

    # Act & Assert - Subsequent calls should not fire
    assert epoch_trigger_at_5.update(epoch=6, metrics={}) is False
    assert epoch_trigger_at_5.update(epoch=7, metrics={}) is False


def test_epoch_trigger_fires_again_after_reset(epoch_trigger_at_5: EpochTransitionTrigger) -> None:
    """Test that epoch trigger fires again after reset.

    Args:
        epoch_trigger_at_5: Trigger configured for epoch 5.
    """
    # Act - Fire once
    assert epoch_trigger_at_5.update(epoch=5, metrics={}) is True

    # Act - Reset
    epoch_trigger_at_5.reset()

    # Act & Assert - Should fire again
    assert epoch_trigger_at_5.update(epoch=5, metrics={}) is True


def test_epoch_trigger_fires_immediately_at_epoch_0(
    epoch_trigger_at_0: EpochTransitionTrigger,
) -> None:
    """Test epoch trigger configured for epoch 0 fires immediately.

    Args:
        epoch_trigger_at_0: Trigger configured for epoch 0.
    """
    # Act & Assert
    assert epoch_trigger_at_0.update(epoch=0, metrics={}) is True


def test_epoch_trigger_state_dict_round_trip(epoch_trigger_at_5: EpochTransitionTrigger) -> None:
    """Test state_dict and load_state_dict preserve trigger state.

    Args:
        epoch_trigger_at_5: Trigger configured for epoch 5.
    """
    # Act - Fire the trigger
    assert epoch_trigger_at_5.update(epoch=5, metrics={}) is True

    # Act - Save state
    state = epoch_trigger_at_5.state_dict()

    # Assert - State contains expected keys
    assert "at_epoch" in state
    assert "fired" in state
    assert state["at_epoch"] == 5
    assert state["fired"] is True

    # Act - Create fresh trigger and load state
    fresh_trigger = EpochTransitionTrigger(at_epoch=0)
    fresh_trigger.load_state_dict(state)

    # Assert - State restored correctly
    assert fresh_trigger._at_epoch == 5
    assert fresh_trigger._fired is True
    assert fresh_trigger.update(epoch=10, metrics={}) is False  # Should not fire (already fired)


def test_epoch_trigger_state_dict_before_fire(epoch_trigger_at_5: EpochTransitionTrigger) -> None:
    """Test state_dict before trigger fires captures unfired state.

    Args:
        epoch_trigger_at_5: Trigger configured for epoch 5.
    """
    # Act - Do not fire, save state
    state = epoch_trigger_at_5.state_dict()

    # Assert
    assert state["fired"] is False
    assert state["at_epoch"] == 5

    # Act - Create fresh trigger and load unfired state
    fresh_trigger = EpochTransitionTrigger(at_epoch=0)
    fresh_trigger.load_state_dict(state)

    # Assert - Should be able to fire
    assert fresh_trigger.update(epoch=5, metrics={}) is True


# ===========================================================================
# PlateauTransitionTrigger tests
# ===========================================================================


def test_plateau_trigger_returns_false_on_first_update(
    plateau_trigger_patience_2: PlateauTransitionTrigger,
) -> None:
    """Test that plateau trigger returns False on first update.

    Args:
        plateau_trigger_patience_2: Plateau trigger with patience=2.
    """
    # Act & Assert - First update initializes best, returns False
    assert plateau_trigger_patience_2.update(epoch=0, metrics={"val_loss": 0.5}) is False


def test_plateau_trigger_fires_after_patience_exhausted(
    plateau_trigger_patience_2: PlateauTransitionTrigger,
) -> None:
    """Test that plateau trigger fires after patience epochs without improvement.

    Args:
        plateau_trigger_patience_2: Plateau trigger with patience=2.
    """
    # Act - Initialize and feed data
    plateau_trigger_patience_2.update(epoch=0, metrics={"val_loss": 0.5})  # best=0.5, counter=0

    # Act - Non-improving value (same value): counter=1
    assert plateau_trigger_patience_2.update(epoch=1, metrics={"val_loss": 0.5}) is False

    # Act - Non-improving value (worse): counter=2, trigger fires (counter >= patience)
    assert plateau_trigger_patience_2.update(epoch=2, metrics={"val_loss": 0.500001}) is True


def test_plateau_trigger_resets_counter_on_improvement(
    plateau_trigger_patience_2: PlateauTransitionTrigger,
) -> None:
    """Test that plateau trigger resets counter when metric improves.

    Args:
        plateau_trigger_patience_2: Plateau trigger with patience=2.
    """
    # Act - Initialize
    plateau_trigger_patience_2.update(epoch=0, metrics={"val_loss": 0.5})  # best=0.5, counter=0

    # Act - Non-improve: counter=1
    plateau_trigger_patience_2.update(epoch=1, metrics={"val_loss": 0.5})

    # Act - Improvement (0.5 - 0.4 > 1e-4): best=0.4, counter=0
    assert plateau_trigger_patience_2.update(epoch=2, metrics={"val_loss": 0.4}) is False

    # Act - Non-improve: counter=1
    assert plateau_trigger_patience_2.update(epoch=3, metrics={"val_loss": 0.4}) is False

    # Act - Non-improve: counter=2, trigger fires
    assert plateau_trigger_patience_2.update(epoch=4, metrics={"val_loss": 0.400001}) is True


def test_plateau_trigger_returns_false_for_missing_metric(
    plateau_trigger_patience_2: PlateauTransitionTrigger,
) -> None:
    """Test that plateau trigger returns False if monitored metric is missing.

    Args:
        plateau_trigger_patience_2: Plateau trigger monitoring "val_loss".
    """
    # Act & Assert - Metric dict without monitored key returns False
    assert plateau_trigger_patience_2.update(epoch=0, metrics={"other_loss": 0.5}) is False


def test_plateau_trigger_mode_max(plateau_trigger_max_mode: PlateauTransitionTrigger) -> None:
    """Test plateau trigger in max mode counts decreasing values as non-improving.

    Args:
        plateau_trigger_max_mode: Plateau trigger with mode="max".
    """
    # Act - Initialize with accuracy 0.8
    plateau_trigger_max_mode.update(epoch=0, metrics={"val_acc": 0.8})  # best=0.8, counter=0

    # Act - Decreasing accuracy: counter=1
    assert plateau_trigger_max_mode.update(epoch=1, metrics={"val_acc": 0.79}) is False

    # Act - Decreasing accuracy: counter=2
    assert plateau_trigger_max_mode.update(epoch=2, metrics={"val_acc": 0.78}) is True


def test_plateau_trigger_min_delta_threshold() -> None:
    """Test that min_delta threshold prevents marginal improvements.

    Tests that changes smaller than min_delta are not counted as improvement.
    """
    # Create trigger with patience=2 for clearer test
    trigger = PlateauTransitionTrigger(
        monitor="val_loss",
        patience=2,
        min_delta=1e-4,
        mode="min",
    )

    # Act - Initialize
    trigger.update(epoch=0, metrics={"val_loss": 0.5000})

    # Act - Change less than min_delta (worse): not improvement, counter=1
    # 0.5000 - 0.50005 = -0.00005, which is < 1e-4, so not improvement
    assert trigger.update(epoch=1, metrics={"val_loss": 0.50005}) is False

    # Act - Change more than min_delta (worse): still not improvement, counter=2
    assert trigger.update(epoch=2, metrics={"val_loss": 0.50011}) is True

    # Now test that change less than min_delta in positive direction is also not counted
    trigger2 = PlateauTransitionTrigger(
        monitor="val_loss",
        patience=2,
        min_delta=1e-4,
        mode="min",
    )
    trigger2.update(epoch=0, metrics={"val_loss": 0.5000})

    # Marginal improvement (0.5000 - 0.49999 = 0.00001 < 1e-4): not counted as improvement
    assert trigger2.update(epoch=1, metrics={"val_loss": 0.49999}) is False
    assert trigger2.update(epoch=2, metrics={"val_loss": 0.49998}) is True


def test_plateau_trigger_state_dict_round_trip(
    plateau_trigger_patience_2: PlateauTransitionTrigger,
) -> None:
    """Test state_dict and load_state_dict preserve plateau state.

    Args:
        plateau_trigger_patience_2: Plateau trigger with patience=2.
    """
    # Act - Update trigger to partial state
    plateau_trigger_patience_2.update(epoch=0, metrics={"val_loss": 0.5})
    plateau_trigger_patience_2.update(epoch=1, metrics={"val_loss": 0.5})  # counter=1

    # Act - Save state
    state = plateau_trigger_patience_2.state_dict()

    # Assert - State contains expected keys
    assert "monitor" in state
    assert "patience" in state
    assert "min_delta" in state
    assert "mode" in state
    assert "best" in state
    assert "counter" in state
    assert state["monitor"] == "val_loss"
    assert state["patience"] == 2
    assert state["best"] == 0.5
    assert state["counter"] == 1

    # Act - Create fresh trigger and load state
    fresh_trigger = PlateauTransitionTrigger(monitor="dummy", patience=999)
    fresh_trigger.load_state_dict(state)

    # Assert - Continue from saved state
    assert fresh_trigger._best == 0.5
    assert fresh_trigger._counter == 1
    assert fresh_trigger.update(epoch=2, metrics={"val_loss": 0.5}) is True  # counter=2, fires


def test_plateau_trigger_reset_clears_state(
    plateau_trigger_patience_2: PlateauTransitionTrigger,
) -> None:
    """Test that reset clears best and counter.

    Args:
        plateau_trigger_patience_2: Plateau trigger with patience=2.
    """
    # Act - Update trigger
    plateau_trigger_patience_2.update(epoch=0, metrics={"val_loss": 0.5})
    plateau_trigger_patience_2.update(epoch=1, metrics={"val_loss": 0.49})

    # Act - Reset
    plateau_trigger_patience_2.reset()

    # Assert - State cleared
    assert plateau_trigger_patience_2._best is None
    assert plateau_trigger_patience_2._counter == 0

    # Act & Assert - Next update reinitializes
    assert plateau_trigger_patience_2.update(epoch=2, metrics={"val_loss": 0.4}) is False
    assert plateau_trigger_patience_2._best == 0.4


# ===========================================================================
# NoTransitionTrigger tests
# ===========================================================================


def test_no_trigger_always_returns_false(no_trigger: NoTransitionTrigger) -> None:
    """Test that NoTransitionTrigger always returns False.

    Args:
        no_trigger: A NoTransitionTrigger instance.
    """
    # Act & Assert - Multiple calls with various epochs and metrics
    assert no_trigger.update(epoch=0, metrics={}) is False
    assert no_trigger.update(epoch=5, metrics={"val_loss": 0.5}) is False
    assert no_trigger.update(epoch=100, metrics={"val_loss": 0.1, "val_acc": 0.99}) is False


def test_no_trigger_state_dict_is_empty(no_trigger: NoTransitionTrigger) -> None:
    """Test that NoTransitionTrigger.state_dict returns empty dict.

    Args:
        no_trigger: A NoTransitionTrigger instance.
    """
    # Act
    state = no_trigger.state_dict()

    # Assert
    assert state == {}
    assert isinstance(state, dict)


def test_no_trigger_load_state_dict_is_noop(no_trigger: NoTransitionTrigger) -> None:
    """Test that NoTransitionTrigger.load_state_dict does not raise.

    Args:
        no_trigger: A NoTransitionTrigger instance.
    """
    # Act & Assert - Should not raise
    no_trigger.load_state_dict({})
    no_trigger.load_state_dict({"extra": "data"})

    # Assert - Still returns False after load
    assert no_trigger.update(epoch=0, metrics={}) is False


def test_no_trigger_reset_is_noop(no_trigger: NoTransitionTrigger) -> None:
    """Test that NoTransitionTrigger.reset does not raise and is a no-op.

    Args:
        no_trigger: A NoTransitionTrigger instance.
    """
    # Act & Assert - Should not raise
    no_trigger.reset()
    no_trigger.reset()

    # Assert - Still returns False
    assert no_trigger.update(epoch=0, metrics={}) is False


# ===========================================================================
# Parametrized tests for multiple scenarios
# ===========================================================================


@pytest.mark.parametrize(
    "at_epoch,test_epochs,expected_results",
    [
        (0, [0, 1], [True, False]),  # Fires at 0, then stays False
        (5, [4, 5, 6], [False, True, False]),  # Doesn't fire at 4, fires at 5, stays False at 6
        (
            10,
            [9, 10, 11],
            [False, True, False],
        ),  # Doesn't fire at 9, fires at 10, stays False at 11
    ],
)
def test_epoch_trigger_various_epochs(
    at_epoch: int,
    test_epochs: list[int],
    expected_results: list[bool],
) -> None:
    """Test epoch trigger with various at_epoch and test_epoch combinations.

    Args:
        at_epoch: Target epoch for trigger.
        test_epochs: List of epochs to test sequentially.
        expected_results: Expected results for each epoch.
    """
    # Act
    trigger = EpochTransitionTrigger(at_epoch=at_epoch)

    # Act & Assert - Update and verify each epoch
    for epoch, expected in zip(test_epochs, expected_results, strict=True):
        result = trigger.update(epoch=epoch, metrics={})
        assert result is expected, (
            f"at_epoch={at_epoch}, epoch={epoch}: expected {expected}, got {result}"
        )


@pytest.mark.parametrize(
    "mode,metric_sequence,expected_fire_epoch",
    [
        # min mode: improvement if (best - current) > min_delta
        # [0.5, 0.5, 0.500001]: epoch 0 init, epoch 1 no improve (counter=1), epoch 2 no improve (counter=2, fires)
        ("min", [0.5, 0.5, 0.500001], 2),
        # max mode: improvement if (current - best) > min_delta
        # [0.5, 0.5, 0.499999]: epoch 0 init, epoch 1 no improve (counter=1), epoch 2 no improve (counter=2, fires)
        ("max", [0.5, 0.5, 0.499999], 2),
    ],
)
def test_plateau_trigger_mode_behavior(
    mode: Literal["min", "max"],
    metric_sequence: list[float],
    expected_fire_epoch: int,
) -> None:
    """Test plateau trigger in different modes with various metric sequences.

    Args:
        mode: "min" or "max" mode.
        metric_sequence: Sequence of metric values.
        expected_fire_epoch: Epoch at which trigger should fire.
    """
    # Act
    trigger = PlateauTransitionTrigger(
        monitor="metric",
        patience=2,
        min_delta=1e-4,
        mode=cast(Literal["min", "max"], mode),
    )

    # Act & Assert - Update with each metric and check fire state
    for epoch, value in enumerate(metric_sequence):
        result = trigger.update(epoch=epoch, metrics={"metric": value})
        if epoch < expected_fire_epoch:
            assert result is False, f"Should not fire at epoch {epoch}"
        elif epoch == expected_fire_epoch:
            assert result is True, f"Should fire at epoch {epoch}"
        else:
            # After firing, subsequent updates should return False
            pass
