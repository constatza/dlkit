"""Read-only projection of optimization state for Lightning integration."""

from __future__ import annotations

from .state import ActiveStage, RunningOptimizerPolicy


class OptimizationMetricsView:
    """Read-only projection of optimization program state.

    Provides learning rate queries and optimizer metadata without coupling
    to Lightning's trainer.optimizers list.

    Attributes:
        _program: The running optimization program (stored by reference).
    """

    def __init__(self, program: RunningOptimizerPolicy) -> None:
        """Initialize the metrics view.

        Args:
            program: The running optimization program to project.
        """
        self._program = program

    def current_learning_rates(self) -> dict[str, float]:
        """Return learning rates from the active stage's optimizer param groups.

        For a ConcurrentOptimizer stage, param_groups covers all sub-optimizers.

        Returns:
            Dict mapping optimizer keys to their learning rates.
        """
        rates: dict[str, float] = {}
        stage: ActiveStage = self._program.current

        for group_idx, group in enumerate(stage.optimizer.param_groups):
            lr = group.get("lr", 0.0)
            key = (
                f"stage_{stage.stage_index}"
                if group_idx == 0
                else f"stage_{stage.stage_index}_group_{group_idx}"
            )
            rates[key] = float(lr)

        return rates

    def optimizer_names(self) -> tuple[str, ...]:
        """Return the name of the currently active stage.

        Returns:
            Tuple containing the stage's name label.
        """
        return (self._program.current.name,)

    def active_stage_index(self) -> int:
        """Return zero-based index of the currently active stage.

        Returns:
            The active_index from the program.
        """
        return self._program.active_index
