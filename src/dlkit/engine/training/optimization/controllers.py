"""Controllers for optimization during Lightning training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

from torch import Tensor, nn
from torch.optim import LBFGS

from .concurrent_optimizer import ConcurrentOptimizer
from .state import RunningOptimizerPolicy
from .state_repository import IOptimizationStateRepository
from .stepping import IStepPolicy, LBFGSStageStepper, StepAllOptimizers

if TYPE_CHECKING:
    from dlkit.infrastructure.config.optimizer_policy import OptimizerPolicySettings


def _requires_manual_optimization(program: RunningOptimizerPolicy) -> bool:
    """Return True when manual optimizer stepping is needed.

    Manual optimization is required when there are 2+ sequential stages (only one
    should step per batch) or when any stage uses LBFGS (needs a closure).

    Args:
        program: The assembled optimization program.

    Returns:
        True if automatic_optimization should be disabled on the Lightning module.
    """
    if len(program.stages) > 1:
        return True
    for stage in program.stages:
        opt = stage.optimizer
        sub_opts = opt.sub_optimizers if isinstance(opt, ConcurrentOptimizer) else [opt]
        if any(isinstance(s, LBFGS) for s in sub_opts):
            return True
    return False


def _pick_step_policy(program: RunningOptimizerPolicy) -> IStepPolicy:
    """Select the manual-stepping policy for the program.

    Args:
        program: The running optimization program.

    Returns:
        An IStepPolicy suited to the program's optimizers.
    """
    for stage in program.stages:
        opt = stage.optimizer
        sub_opts = opt.sub_optimizers if isinstance(opt, ConcurrentOptimizer) else [opt]
        if any(isinstance(s, LBFGS) for s in sub_opts):
            return LBFGSStageStepper()
    return StepAllOptimizers()


def build_optimization_controller(
    model: nn.Module,
    optimizer_policy_settings: OptimizerPolicySettings,
) -> IOptimizationController:
    """Build the appropriate optimization controller for a Lightning wrapper.

    Args:
        model: The nn.Module whose parameters will be optimized.
        optimizer_policy_settings: Optimizer/scheduler policy configuration.

    Returns:
        An IOptimizationController configured for the program.
    """
    from .builder import OptimizerPolicyBuilder
    from .state_repository import OptimizationStateRepository

    program = OptimizerPolicyBuilder().build(model, optimizer_policy_settings)
    repository = OptimizationStateRepository()
    if _requires_manual_optimization(program):
        return ManualOptimizationController(program, repository, _pick_step_policy(program))
    return AutomaticOptimizationController(program, repository)


class IOptimizationController(ABC):
    """Abstract interface for optimization control during training."""

    @property
    @abstractmethod
    def requires_manual_optimization(self) -> bool:
        """Check if this controller requires manual optimizer stepping.

        Returns:
            True if trainer.automatic_optimization should be False.
        """
        ...

    @abstractmethod
    def configure_optimizers(self) -> dict[str, object] | list[object]:
        """Return optimizer configuration for Lightning.

        Returns:
            Lightning-compatible optimizer/scheduler dict or list.
        """
        ...

    @abstractmethod
    def manual_step(self, loss_fn: Callable[[], Tensor]) -> Tensor:
        """Execute one manual optimizer step.

        Args:
            loss_fn: Callable that computes and returns the loss tensor.

        Returns:
            The computed loss tensor.
        """
        ...

    @abstractmethod
    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Handle end-of-epoch logic including trigger evaluation and stage transition.

        Args:
            epoch: The epoch number (0-indexed).
            metrics: Dict of current metrics for trigger evaluation.
        """
        ...

    @abstractmethod
    def current_learning_rates(self) -> dict[str, float]:
        """Return learning rates for all currently active optimizers.

        Returns:
            Dict mapping stage keys to their current learning rates.
        """
        ...

    @abstractmethod
    def state_dict(self) -> dict[str, object]:
        """Return controller state for checkpointing.

        Returns:
            State dict suitable for load_state_dict().
        """
        ...

    @abstractmethod
    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore controller state from checkpoint.

        Args:
            state: State dict from state_dict().
        """
        ...


class AutomaticOptimizationController(IOptimizationController):
    """Controller for automatic optimization (Lightning-managed stepping).

    Attributes:
        _program: The running optimization program.
        _repository: Repository for state checkpoint/restore.
    """

    def __init__(
        self,
        program: RunningOptimizerPolicy,
        repository: IOptimizationStateRepository,
    ) -> None:
        """Initialize the automatic optimization controller.

        Args:
            program: The running optimization program.
            repository: Repository for saving/loading state.
        """
        self._program = program
        self._repository = repository

    @property
    def requires_manual_optimization(self) -> bool:
        """Return False (automatic optimization is enabled).

        Returns:
            Always False.
        """
        return False

    def configure_optimizers(self) -> dict[str, object] | list[object]:
        """Configure optimizers and schedulers for Lightning.

        Returns:
            Dict with "optimizer" and optional "lr_scheduler" for a single stage,
            or a list of per-stage dicts when multiple sequential stages are registered.
        """
        stages = self._program.stages

        if len(stages) == 1:
            stage = stages[0]
            config: dict[str, object] = {"optimizer": stage.optimizer}
            if stage.scheduler is not None:
                config["lr_scheduler"] = {
                    "scheduler": stage.scheduler,
                    "monitor": stage.scheduler_monitor,
                    "frequency": stage.scheduler_frequency,
                }
            return config

        entries: list[object] = []
        for stage in stages:
            entry: dict[str, object] = {"optimizer": stage.optimizer}
            if stage.scheduler is not None:
                entry["lr_scheduler"] = {
                    "scheduler": stage.scheduler,
                    "monitor": stage.scheduler_monitor,
                    "frequency": stage.scheduler_frequency,
                }
            entries.append(entry)
        return entries

    def manual_step(self, loss_fn: Callable[[], Tensor]) -> Tensor:
        """Raise error (not used in automatic mode).

        Args:
            loss_fn: Unused.

        Raises:
            RuntimeError: Always raised.
        """
        raise RuntimeError("AutomaticOptimizationController does not support manual stepping")

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Evaluate triggers and advance to next stage if needed.

        Args:
            epoch: Current epoch (0-indexed).
            metrics: Dict of current metrics.
        """
        current = self._program.current
        if current.trigger.update(epoch, metrics):
            current.trigger.reset()
            self._program.advance()

    def current_learning_rates(self) -> dict[str, float]:
        """Return learning rates for all currently active optimizers.

        Returns:
            Dict mapping stage keys to their current learning rates.
        """
        from .metrics import OptimizationMetricsView

        return OptimizationMetricsView(self._program).current_learning_rates()

    def state_dict(self) -> dict[str, object]:
        """Return state for checkpointing.

        Returns:
            State dict from repository.
        """
        return self._repository.save(self._program)

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore state from checkpoint.

        Args:
            state: State dict from state_dict().
        """
        self._repository.restore(self._program, state)


class ManualOptimizationController(IOptimizationController):
    """Controller for manual optimization stepping.

    Used when optimizer stepping must be controlled manually (e.g. LBFGS,
    sequential multi-stage). Requires automatic_optimization=False in Lightning.

    Attributes:
        _program: The running optimization program.
        _repository: Repository for state checkpoint/restore.
        _step_policy: Policy for stepping optimizers.
    """

    def __init__(
        self,
        program: RunningOptimizerPolicy,
        repository: IOptimizationStateRepository,
        step_policy: IStepPolicy,
    ) -> None:
        """Initialize the manual optimization controller.

        Args:
            program: The running optimization program.
            repository: Repository for saving/loading state.
            step_policy: Policy for stepping (e.g., StepAllOptimizers).
        """
        self._program = program
        self._repository = repository
        self._step_policy = step_policy

    @property
    def requires_manual_optimization(self) -> bool:
        """Return True (manual optimization is required).

        Returns:
            Always True.
        """
        return True

    def configure_optimizers(self) -> dict[str, object] | list[object]:
        """Configure optimizers for Lightning (manual mode returns list).

        Returns:
            List of all stage optimizers.
        """
        return [stage.optimizer for stage in self._program.stages]

    def manual_step(self, loss_fn: Callable[[], Tensor]) -> Tensor:
        """Execute one optimizer step using the step policy.

        Args:
            loss_fn: Callable that computes the loss.

        Returns:
            The computed loss tensor.
        """
        return self._step_policy.step(self._program.current, loss_fn)

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Evaluate triggers and advance to next stage if needed.

        Args:
            epoch: Current epoch (0-indexed).
            metrics: Dict of current metrics.
        """
        current = self._program.current
        if current.trigger.update(epoch, metrics):
            current.trigger.reset()
            self._program.advance()

    def current_learning_rates(self) -> dict[str, float]:
        """Return learning rates for all currently active optimizers.

        Returns:
            Dict mapping stage keys to their current learning rates.
        """
        from .metrics import OptimizationMetricsView

        return OptimizationMetricsView(self._program).current_learning_rates()

    def state_dict(self) -> dict[str, object]:
        """Return state for checkpointing.

        Returns:
            State dict from repository.
        """
        return self._repository.save(self._program)

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore state from checkpoint.

        Args:
            state: State dict from state_dict().
        """
        self._repository.restore(self._program, state)
