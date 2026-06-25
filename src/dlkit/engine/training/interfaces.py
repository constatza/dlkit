"""Clean, segregated interfaces for training execution following ISP."""

from __future__ import annotations

from abc import ABC, abstractmethod

from dlkit.common import OptimizationResult, TrainingResult
from dlkit.engine.training.components import RuntimeComponents


class ITrainingExecutor(ABC):
    """Core training execution interface - single responsibility."""

    @abstractmethod
    def execute(self, components: RuntimeComponents, settings: object) -> TrainingResult:
        """Execute training and return TrainingResult, or raise WorkflowError.

        Args:
            components: Pre-built training components (model, trainer, datamodule)
            settings: Global training settings (JobConfig or old-style WorkflowConfig)

        Returns:
            TrainingResult containing metrics, artifacts, and model state

        Raises:
            WorkflowError: If training execution fails
        """
        raise NotImplementedError


class IOptimizationStrategy(ABC):
    """Abstract optimization strategy that can produce OptimizationResult.

    This interface bridges the gap between training executors that return
    TrainingResult and the need for OptimizationResult in optimization workflows.
    Follows ISP by providing a focused interface for optimization concerns.
    """

    @abstractmethod
    def execute_optimization(self, settings: object) -> OptimizationResult:
        """Execute optimization workflow.

        Args:
            settings: Configuration settings with optimization parameters.

        Returns:
            Optimization result with best trial and training result

        Raises:
            WorkflowError: If optimization fails
        """
        raise NotImplementedError
