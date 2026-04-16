"""Read-only projection of optimization state for Lightning integration."""

from __future__ import annotations

from .state import ActiveConcurrentGroup, ActiveStage, RunningOptimizationProgram


def _flatten_active_stages(
    current: ActiveStage | ActiveConcurrentGroup,
) -> tuple[ActiveStage, ...]:
    """Extract all active stages from the current stage/group.

    Args:
        current: The active stage or concurrent group.

    Returns:
        Tuple of all active ActiveStage objects.
    """
    if isinstance(current, ActiveStage):
        return (current,)
    return current.stages


class OptimizationMetricsView:
    """Read-only projection of optimization program state.

    Provides learning rate queries and optimizer metadata without coupling
    to Lightning's trainer.optimizers list. Used by metrics callbacks
    to track optimization progress.

    Attributes:
        _program: The running optimization program (stored by reference).
    """

    def __init__(self, program: RunningOptimizationProgram) -> None:
        """Initialize the metrics view.

        Args:
            program: The running optimization program to project.
        """
        self._program = program

    def current_learning_rates(self) -> dict[str, float]:
        """Return learning rates from all active optimizers.

        Keys are formatted as "stage_{stage_idx}_group_{group_idx}" or
        "stage_{stage_idx}" for single-optimizer stages.

        Returns:
            Dict mapping optimizer keys to their learning rates.
        """
        rates: dict[str, float] = {}
        active_stages = _flatten_active_stages(self._program.current)

        for stage in active_stages:
            # Optimizers have param_groups; each group has an "lr" key
            for group_idx, group in enumerate(stage.optimizer.param_groups):
                lr = group.get("lr", 0.0)
                if len(active_stages) == 1 and group_idx == 0:
                    # Single optimizer, single group: use simple key
                    key = f"stage_{stage.stage_index}"
                else:
                    # Multiple stages or groups: use detailed key
                    key = f"stage_{stage.stage_index}_group_{group_idx}"
                rates[key] = float(lr)

        return rates

    def optimizer_names(self) -> tuple[str, ...]:
        """Return names or labels of active optimizers.

        Returns:
            Tuple of optimizer labels (empty string for unlabeled stages).
        """
        active_stages = _flatten_active_stages(self._program.current)
        return tuple(stage.name for stage in active_stages)

    def active_stage_index(self) -> int:
        """Return zero-based index of the currently active stage.

        Returns:
            The active_index from the program.
        """
        return self._program.active_index
