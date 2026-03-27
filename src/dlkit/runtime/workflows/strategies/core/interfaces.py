"""Clean, segregated interfaces for training execution following ISP."""

from __future__ import annotations

from abc import ABC, abstractmethod

from dlkit.domain import OptimizationResult, TrainingResult
from dlkit.runtime.workflows.factories.build_factory import BuildComponents
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.workflow_configs import OptimizationWorkflowConfig, TrainingWorkflowConfig

# Settings union accepted by training and optimization strategies
WorkflowSettings = GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig


class ITrainingExecutor(ABC):
    """Core training execution interface - single responsibility."""

    @abstractmethod
    def execute(self, components: BuildComponents, settings: WorkflowSettings) -> TrainingResult:
        """Execute training and return TrainingResult, or raise WorkflowError.

        Args:
            components: Pre-built training components (model, trainer, datamodule)
            settings: Global training settings

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
    def execute_optimization(self, settings: WorkflowSettings) -> OptimizationResult:
        """Execute optimization workflow.

        Args:
            settings: Configuration settings with optimization parameters

        Returns:
            Optimization result with best trial and training result

        Raises:
            WorkflowError: If optimization fails
        """
        raise NotImplementedError
