"""Adapter delegating workflow execution to engine entrypoints."""

from __future__ import annotations

from dlkit.common import OptimizationResult, TrainingResult
from dlkit.common.hooks import LifecycleHooks
from dlkit.common.results import ChildOutcome, ConvergenceResult, MultiRunResult, WorkflowResult
from dlkit.engine.workflows.entrypoints import (
    MultiRunSpec,
    apply_mlflow_flag,
)
from dlkit.engine.workflows.entrypoints import (
    converge as runtime_converge,
)
from dlkit.engine.workflows.entrypoints import (
    execute as runtime_execute,
)
from dlkit.engine.workflows.entrypoints import (
    optimize as runtime_optimize,
)
from dlkit.engine.workflows.entrypoints import (
    run_multirun as runtime_run_multirun,
)
from dlkit.engine.workflows.entrypoints import (
    run_multirun_spec as runtime_run_multirun_spec,
)
from dlkit.engine.workflows.entrypoints import (
    train as runtime_train,
)
from dlkit.engine.workflows.entrypoints._settings import WorkflowSettings
from dlkit.infrastructure.config.job_config import (
    ConvergenceJobConfig,
    MultiRunJobConfig,
    SearchJobConfig,
    TrainingJobConfig,
)
from dlkit.interfaces.api.domain.override_types import (
    ConvergenceOverrides,
    ExecutionOverrides,
    OptimizationOverrides,
    TrainingOverrides,
)


class EngineWorkflowExecutor:
    """Concrete executor adapter for engine workflow entrypoints."""

    def converge(
        self,
        settings: WorkflowSettings,
        overrides: ConvergenceOverrides | None = None,
        *,
        mlflow: bool = False,
    ) -> ConvergenceResult:
        """Execute a convergence study via engine entrypoint.

        Args:
            settings: Convergence workflow configuration settings.
            overrides: Optional convergence overrides.
            mlflow: If True, ensure MLflow tracking is configured.

        Returns:
            ConvergenceResult with convergence points and tracking metadata.

        Raises:
            TypeError: If settings is not a ConvergenceJobConfig.
        """
        settings_with_tracking = apply_mlflow_flag(settings, mlflow)
        if not isinstance(settings_with_tracking, ConvergenceJobConfig):
            raise TypeError(
                f"converge() requires ConvergenceJobConfig, got {type(settings_with_tracking).__name__}"
            )
        return runtime_converge(settings_with_tracking, overrides=overrides)

    def train(
        self,
        settings: WorkflowSettings,
        overrides: TrainingOverrides | None = None,
        *,
        mlflow: bool = False,
        hooks: LifecycleHooks | None = None,
    ) -> TrainingResult:
        """Execute a training workflow via engine entrypoint.

        Args:
            settings: Training workflow configuration settings.
            overrides: Optional training overrides.
            mlflow: If True, ensure MLflow tracking is configured.
            hooks: Optional lifecycle hooks for training events.

        Returns:
            TrainingResult containing trained model state and metrics.

        Raises:
            TypeError: If settings is not a TrainingJobConfig.
        """
        settings_with_tracking = apply_mlflow_flag(settings, mlflow)
        if not isinstance(settings_with_tracking, TrainingJobConfig):
            raise TypeError(
                f"train() requires TrainingJobConfig, got {type(settings_with_tracking).__name__}"
            )
        return runtime_train(settings_with_tracking, overrides=overrides, hooks=hooks)

    def optimize(
        self,
        settings: WorkflowSettings,
        overrides: OptimizationOverrides | None = None,
        *,
        mlflow: bool = False,
        hooks: LifecycleHooks | None = None,
    ) -> OptimizationResult:
        """Execute optimization via engine entrypoint.

        Args:
            settings: Optimization workflow configuration settings.
            overrides: Optional optimization overrides.
            mlflow: If True, ensure MLflow tracking is configured.
            hooks: Optional lifecycle hooks for training events.

        Returns:
            OptimizationResult containing best model and trial history.

        Raises:
            TypeError: If settings is not a SearchJobConfig.
        """
        settings_with_tracking = apply_mlflow_flag(settings, mlflow)
        if not isinstance(settings_with_tracking, SearchJobConfig):
            raise TypeError(
                f"optimize() requires SearchJobConfig, got {type(settings_with_tracking).__name__}"
            )
        return runtime_optimize(settings_with_tracking, overrides=overrides, hooks=hooks)

    def run_multirun_config(
        self,
        settings: MultiRunJobConfig,
        *,
        mlflow: bool = False,
    ) -> MultiRunResult[ChildOutcome[WorkflowResult]]:
        """Execute a multirun sweep from a validated MultiRunJobConfig.

        Args:
            settings: Validated multirun job configuration.
            mlflow: Accepted for signature symmetry with the other executor
                methods; has no effect. A multirun sweep's entire purpose is
                parent/child MLflow linkage, so the engine entrypoint always
                configures tracking from the first child's own settings
                (mlflow-flag-forced) regardless of this flag — see
                ``dlkit.engine.workflows.entrypoints.multirun._run_sweep``.

        Returns:
            MultiRunResult with the parent run id, tracking URI, and one
            ChildOutcome per child, in expansion order.

        Raises:
            TypeError: If settings is not a MultiRunJobConfig.
        """
        if not isinstance(settings, MultiRunJobConfig):
            raise TypeError(
                f"run_multirun_config() requires MultiRunJobConfig, got {type(settings).__name__}"
            )
        return runtime_run_multirun(settings)

    def run_multirun_spec(
        self,
        spec: MultiRunSpec,
        *,
        mlflow: bool = False,
    ) -> MultiRunResult[ChildOutcome[WorkflowResult]]:
        """Execute a multirun sweep from an already-built MultiRunSpec.

        Args:
            spec: Fully-specified sweep: parent identity plus expanded children.
            mlflow: Accepted for signature symmetry with the other executor
                methods; has no effect — see ``run_multirun_config``'s docstring.

        Returns:
            MultiRunResult with the parent run id, tracking URI, and one
            ChildOutcome per child, in expansion order.
        """
        return runtime_run_multirun_spec(spec)

    def execute(
        self,
        settings: WorkflowSettings,
        overrides: ExecutionOverrides | None = None,
        *,
        mlflow: bool = False,
        hooks: LifecycleHooks | None = None,
    ) -> TrainingResult | OptimizationResult | ConvergenceResult:
        """Execute a workflow with intelligent routing via engine entrypoint.

        Args:
            settings: Workflow configuration settings.
            overrides: Optional execution overrides.
            mlflow: If True, ensure MLflow tracking is configured.
            hooks: Optional lifecycle hooks for training events.

        Returns:
            TrainingResult or OptimizationResult depending on workflow type.
        """
        settings_with_tracking = apply_mlflow_flag(settings, mlflow)
        return runtime_execute(settings_with_tracking, overrides=overrides, hooks=hooks)
