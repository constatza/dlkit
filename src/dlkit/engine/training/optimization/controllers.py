"""Controllers for optimization during Lightning training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

from torch import Tensor, nn

from .state import ActiveConcurrentGroup, ActiveStage, RunningOptimizerPolicy
from .state_repository import IOptimizationStateRepository
from .stepping import IStepPolicy, LBFGSStageStepper, StepAllOptimizers

if TYPE_CHECKING:
    from dlkit.infrastructure.config.optimizer_policy import OptimizerPolicySettings


def _flatten_all_stages(program: RunningOptimizerPolicy) -> tuple[ActiveStage, ...]:
    """Extract all stages from every entry in a program, flattening concurrent groups.

    Args:
        program: The running optimization program.

    Returns:
        Tuple of all ActiveStage objects across all entries.
    """
    stages: list[ActiveStage] = []
    for entry in program.stages:
        if isinstance(entry, ActiveStage):
            stages.append(entry)
        elif isinstance(entry, ActiveConcurrentGroup):
            stages.extend(entry.stages)
    return tuple(stages)


def _requires_manual_optimization(program: RunningOptimizerPolicy) -> bool:
    """Return True when manual optimizer stepping is needed.

    Manual optimization is required when any stage uses LBFGS (needs a closure)
    or when there are multiple sequential stages (only one should step per batch).

    Args:
        program: The assembled optimization program.

    Returns:
        True if automatic_optimization should be disabled on the Lightning module.
    """
    sequential_count = sum(1 for e in program.stages if isinstance(e, ActiveStage))
    if sequential_count > 1:
        return True
    for entry in program.stages:
        stages = [entry] if isinstance(entry, ActiveStage) else list(entry.stages)
        for stage in stages:
            if any(x in stage.optimizer.__class__.__name__.lower() for x in ("lbfgs", "manual")):
                return True
    return False


def _pick_step_policy(program: RunningOptimizerPolicy) -> IStepPolicy:
    """Select the manual-stepping policy for the program.

    Uses LBFGSStageStepper when any stage holds an LBFGS-family optimizer.
    Falls back to StepAllOptimizers otherwise.

    Args:
        program: The running optimization program.

    Returns:
        An IStepPolicy suited to the program's optimizers.
    """
    for entry in program.stages:
        stages = [entry] if isinstance(entry, ActiveStage) else list(entry.stages)
        for stage in stages:
            if "lbfgs" in stage.optimizer.__class__.__name__.lower():
                return LBFGSStageStepper()
    return StepAllOptimizers()


def build_optimization_controller(
    model: nn.Module,
    optimizer_policy_settings: OptimizerPolicySettings,
) -> IOptimizationController:
    """Build the appropriate optimization controller for a Lightning wrapper.

    Constructs the optimizer program from settings, then picks between automatic
    and manual optimization based on the program's requirements.

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
    """Abstract interface for optimization control during training.

    Controllers bridge the optimization program and Lightning's training loop,
    handling optimizer registration, manual vs automatic stepping, and
    stage transitions.
    """

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

        Called only when requires_manual_optimization is True.
        The wrapper is responsible for zero_grad and backward.

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

    Uses Lightning's automatic optimization mode. Optimizers and schedulers
    are registered with Lightning and stepped automatically.

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
            Dict with "optimizer" and optional "lr_scheduler" keys for a
            single optimizer, or a list of per-optimizer dicts when multiple
            concurrent stages are registered.
        """
        all_stages = _flatten_all_stages(self._program)

        if len(all_stages) == 1:
            stage = all_stages[0]
            config: dict[str, object] = {"optimizer": stage.optimizer}
            if stage.scheduler is not None:
                config["lr_scheduler"] = {
                    "scheduler": stage.scheduler,
                    "monitor": stage.scheduler_monitor,
                    "frequency": stage.scheduler_frequency,
                }
            return config

        # Multiple concurrent optimizers: each entry preserves its scheduler.
        entries: list[object] = []
        for stage in all_stages:
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

    Used when optimizer stepping must be controlled manually (e.g., LBFGS,
    alternating optimizers). Requires automatic_optimization=False in Lightning.

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
            step_policy: Policy for stepping (e.g., StepAllOptimizers, AlternatingStepPolicy).
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
        """Configure optimizers for Lightning.

        Returns:
            List of all optimizers (Lightning requires this for manual mode).
        """
        return [stage.optimizer for stage in _flatten_all_stages(self._program)]

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
