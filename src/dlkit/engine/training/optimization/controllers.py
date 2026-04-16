"""Controllers for optimization during Lightning training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from torch import Tensor

from .state import ActiveConcurrentGroup, ActiveStage, RunningOptimizationProgram
from .state_repository import IOptimizationStateRepository
from .stepping import IStepPolicy


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
        program: RunningOptimizationProgram,
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
            Dict with "optimizer" and optional "lr_scheduler" keys,
            or list of optimizers if multiple exist.
        """
        # Flatten all stages to get optimizers
        all_stages = self._flatten_all_stages()

        if len(all_stages) == 1:
            # Single optimizer: return dict format
            stage = all_stages[0]
            config: dict[str, object] = {"optimizer": stage.optimizer}

            if stage.scheduler is not None:
                config["lr_scheduler"] = {
                    "scheduler": stage.scheduler,
                    "monitor": stage.scheduler_monitor,
                    "frequency": stage.scheduler_frequency,
                }

            return config

        else:
            # Multiple optimizers: return list format
            return [stage.optimizer for stage in all_stages]

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

    def _flatten_all_stages(self) -> tuple[ActiveStage, ...]:
        """Extract all stages from program, flattening concurrent groups.

        Returns:
            Tuple of all ActiveStage objects.
        """
        stages: list[ActiveStage] = []
        for entry in self._program.stages:
            if isinstance(entry, ActiveStage):
                stages.append(entry)
            elif isinstance(entry, ActiveConcurrentGroup):
                stages.extend(entry.stages)
        return tuple(stages)


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
        program: RunningOptimizationProgram,
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
        all_stages = self._flatten_all_stages()
        return [stage.optimizer for stage in all_stages]

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

    def _flatten_all_stages(self) -> tuple[ActiveStage, ...]:
        """Extract all stages from program, flattening concurrent groups.

        Returns:
            Tuple of all ActiveStage objects.
        """
        stages: list[ActiveStage] = []
        for entry in self._program.stages:
            if isinstance(entry, ActiveStage):
                stages.append(entry)
            elif isinstance(entry, ActiveConcurrentGroup):
                stages.extend(entry.stages)
        return tuple(stages)
