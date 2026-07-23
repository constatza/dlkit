"""Core API functions for training, optimization, and inference."""

from __future__ import annotations

from pathlib import Path

from lightning.pytorch import LightningDataModule

from dlkit.common import (
    OptimizationResult,
    TrainingResult,
)
from dlkit.common.hooks import LifecycleHooks
from dlkit.common.results import (
    ChildOutcome,
    ConvergenceResult,
    FailurePolicy,
    MultiRunResult,
    WorkflowResult,
)
from dlkit.engine.workflows.entrypoints import (
    ExistingRunsSource,
    MultiRunSpec,
    expand_child_sources,
)
from dlkit.engine.workflows.entrypoints._settings import WorkflowSettings
from dlkit.engine.workflows.factories.inference_data_factory import (
    build_inference_datamodule as _build_inference_datamodule,
)
from dlkit.infrastructure.config.job_config import InferenceJobConfig, MultiRunJobConfig
from dlkit.interfaces.api.adapters import EngineWorkflowExecutor
from dlkit.interfaces.api.domain.override_types import (
    ConvergenceOverrides,
    OptimizationOverrides,
    TrainingOverrides,
)

_DEFAULT_EVALUATE_MULTIRUN_EXPERIMENT_NAME = "dlkit-evaluate"

_executor: EngineWorkflowExecutor = EngineWorkflowExecutor()


def train(
    settings: WorkflowSettings,
    overrides: TrainingOverrides | None = None,
    *,
    mlflow: bool = False,
) -> TrainingResult:
    """Run training with optional overrides.

    Args:
        settings: Training workflow configuration settings.
        overrides: Optional training overrides (paths coerced to Path objects).
        mlflow: If True, force MLflow tracking behavior for this execution.

    Returns:
        TrainingResult containing trained model state and metrics.
    """
    return _executor.train(
        settings,
        overrides=overrides,
        mlflow=mlflow,
    )


def build_inference_datamodule(
    settings: InferenceJobConfig,
    *,
    checkpoint_override: Path | str | None = None,
) -> LightningDataModule:
    """Build a datamodule for inference batch iteration.

    No training wrapper, no loss, no optimizer. Only run/experiment, data sections.
    Pure function: no class, no side effects beyond datamodule construction.

    Args:
        settings: Inference job configuration with data section.
        checkpoint_override: Checkpoint path supplied directly by the caller,
            used to auto-locate a colocated split file when
            ``data.splits.filepath`` is unset. Takes precedence over
            ``settings.model.checkpoint``.

    Returns:
        Configured LightningDataModule ready for predict_dataloader iteration.

    Raises:
        ValueError: If data section is not configured.
    """
    return _build_inference_datamodule(settings, checkpoint_override=checkpoint_override)


# REMOVED: Old infer() and predict_with_config() functions
# Use the new load_model() API instead:
#
#   from dlkit import load_model
#   predictor = load_model("model.ckpt", device="cuda")
#   result = predictor.predict(inputs)
#
# See documentation for migration guide.


def converge(
    settings: WorkflowSettings,
    overrides: ConvergenceOverrides | None = None,
    *,
    mlflow: bool = False,
) -> ConvergenceResult:
    """Run a sample-size convergence study with optional overrides.

    Args:
        settings: Convergence workflow configuration settings.
        overrides: Optional convergence overrides (sizes, repeats, target).
        mlflow: If True, force MLflow tracking behavior for this execution.

    Returns:
        ConvergenceResult with convergence points, n_star, and tracking metadata.
    """
    return _executor.converge(
        settings,
        overrides=overrides,
        mlflow=mlflow,
    )


def run_multirun_config(
    settings: MultiRunJobConfig,
    *,
    mlflow: bool = False,
    hooks: LifecycleHooks | None = None,
) -> MultiRunResult[ChildOutcome[WorkflowResult]]:
    """Run a multirun sweep from a validated MultiRunJobConfig.

    Args:
        settings: Validated multirun job configuration.
        mlflow: Accepted for signature symmetry with other workflow functions;
            has no effect — a multirun sweep always configures MLflow tracking
            (parent/child linkage is the point of a sweep). See
            ``EngineWorkflowExecutor.run_multirun_config``'s docstring.
        hooks: Optional lifecycle hooks fired around the parent run and
            forwarded into every child's own workflow execution.

    Returns:
        MultiRunResult with the parent run id, tracking URI, and one
        ChildOutcome per child, in expansion order.
    """
    return _executor.run_multirun_config(settings, mlflow=mlflow, hooks=hooks)


def run_multirun_spec(
    spec: MultiRunSpec,
    *,
    mlflow: bool = False,
    hooks: LifecycleHooks | None = None,
) -> MultiRunResult[ChildOutcome[WorkflowResult]]:
    """Run a multirun sweep from an already-built MultiRunSpec.

    Args:
        spec: Fully-specified sweep: parent identity plus expanded children.
        mlflow: Accepted for signature symmetry with other workflow functions;
            has no effect — see ``run_multirun_config``'s docstring.
        hooks: Optional lifecycle hooks fired around the parent run and
            forwarded into every child's own workflow execution.

    Returns:
        MultiRunResult with the parent run id, tracking URI, and one
        ChildOutcome per child, in expansion order.
    """
    return _executor.run_multirun_spec(spec, mlflow=mlflow, hooks=hooks)


def evaluate_multirun(
    settings: InferenceJobConfig,
    parent_run_id: str,
    *,
    failure_policy: FailurePolicy = "fail_fast",
    hooks: LifecycleHooks | None = None,
) -> MultiRunResult[ChildOutcome[WorkflowResult]]:
    """Batch-evaluate every active child run of an existing parent run.

    Replaces the old bespoke ``evaluate_multirun()`` fan-out: builds an
    ``ExistingRunsSource`` (one evaluate child per active child of
    ``parent_run_id``, each pointed at that child's own checkpoint) and runs
    it through the same ``run_multirun_spec()`` every other sweep uses — a
    new "evaluate sweep" parent run is opened and tagged
    ``multirun.source_parent_run_id`` for traceability back to the run being
    evaluated, and every child is discoverable via the standard
    ``mlflow.parentRunId`` mechanism, same as any other sweep.

    Args:
        settings: Inference job configuration shared across all children;
            each child's ``model.checkpoint`` is overridden to that child's
            own run.
        parent_run_id: MLflow run id whose active children should each be
            evaluated.
        failure_policy: How to react when a child raises. ``"fail_fast"``
            (default) propagates immediately, matching the old
            ``evaluate_multirun()``'s all-or-nothing behavior exactly.
            ``"continue"``/``"continue_mark_parent_failed"`` are also
            available.
        hooks: Optional lifecycle hooks fired around the parent run and
            forwarded into every child's own evaluate() call.

    Returns:
        MultiRunResult with the new evaluate-sweep parent run id, tracking
        URI, and one ChildOutcome per evaluated child, in expansion order.
    """
    source = ExistingRunsSource(parent_run_id=parent_run_id, settings=settings)
    children = expand_child_sources([source])
    experiment_name = (
        settings.experiment.name
        if settings.experiment
        else _DEFAULT_EVALUATE_MULTIRUN_EXPERIMENT_NAME
    )
    spec = MultiRunSpec(
        experiment_name=experiment_name,
        parent_run_name=f"evaluate-{parent_run_id}",
        parent_tags={"multirun.source_parent_run_id": parent_run_id},
        failure_policy=failure_policy,
        children=children,
    )
    return _executor.run_multirun_spec(spec, hooks=hooks)


def optimize(
    settings: WorkflowSettings,
    overrides: OptimizationOverrides | None = None,
    *,
    mlflow: bool = False,
) -> OptimizationResult:
    """Run Optuna hyperparameter optimization with optional overrides.

    Args:
        settings: Optimization workflow configuration settings.
        overrides: Optional optimization overrides (paths coerced to Path objects).
        mlflow: If True, force MLflow tracking behavior for this execution.

    Returns:
        OptimizationResult containing best model and trial history.
    """
    return _executor.optimize(
        settings,
        overrides=overrides,
        mlflow=mlflow,
    )
