"""Runtime-owned convergence workflow entrypoint."""

from __future__ import annotations

from dlkit.common.errors import WorkflowError
from dlkit.common.hooks import LifecycleHooks
from dlkit.common.results import ConvergenceResult
from dlkit.engine.tracking.mlflow_tracker import MLflowTracker
from dlkit.engine.workflows.convergence.orchestrator import ConvergenceOrchestrator
from dlkit.engine.workflows.multi_run import MultiRunOrchestrator
from dlkit.infrastructure.config.job_config import ConvergenceJobConfig
from dlkit.infrastructure.utils.logging_config import get_logger

from ._entrypoint_context import EntrypointContext
from ._override_types import ConvergenceOverrides, require_override_model

logger = get_logger(__name__)


def _apply_convergence_overrides(
    settings: ConvergenceJobConfig,
    overrides: ConvergenceOverrides,
) -> ConvergenceJobConfig:
    """Apply convergence-specific overrides to settings.

    Convergence fields (sizes, repeats, target) are patched into the
    ``convergence`` section. Experiment/run-name/tags are patched via
    the ``experiment`` section.

    Args:
        settings: Validated convergence job configuration.
        overrides: Runtime overrides to apply.

    Returns:
        New ConvergenceJobConfig with all overrides applied.
    """
    patch: dict[str, object] = {}
    convergence_patch: dict[str, object] = {}

    if overrides.sizes is not None:
        convergence_patch["sizes"] = overrides.sizes
    if overrides.repeats is not None:
        convergence_patch["repeats"] = overrides.repeats
    if overrides.target is not None:
        convergence_patch["target"] = overrides.target
    if convergence_patch:
        patch["convergence"] = convergence_patch

    experiment_patch: dict[str, object] = {}
    if overrides.experiment_name is not None:
        experiment_patch["name"] = overrides.experiment_name
    if overrides.run_name is not None:
        experiment_patch["run_name"] = overrides.run_name
    if overrides.tags is not None:
        experiment_patch["tags"] = overrides.tags
    if experiment_patch:
        patch["experiment"] = experiment_patch

    return settings.patch(patch) if patch else settings


def converge(
    settings: ConvergenceJobConfig,
    overrides: ConvergenceOverrides | None = None,
    *,
    hooks: LifecycleHooks | None = None,
) -> ConvergenceResult:
    """Run a convergence study through runtime orchestration.

    Args:
        settings: Validated convergence job configuration.
        overrides: Optional convergence-specific runtime overrides.
        hooks: Optional lifecycle hooks fired around each nested run.

    Returns:
        ConvergenceResult with one ConvergencePoint per evaluated size, n_star,
        total duration, and MLflow tracking metadata when tracking is enabled.

    Raises:
        WorkflowError: If the convergence study fails for any reason.
    """
    logger.info(
        "Convergence | repeats={} target={}",
        getattr(getattr(settings, "convergence", None), "repeats", "?"),
        getattr(getattr(settings, "convergence", None), "target", "?"),
    )
    validated_overrides = require_override_model(overrides, ConvergenceOverrides)
    context = EntrypointContext.prepare(
        settings,
        validated_overrides,
        workflow_name="convergence",
    )

    def run_convergence() -> ConvergenceResult:
        convergence_settings = context.settings
        if not isinstance(convergence_settings, ConvergenceJobConfig):
            raise WorkflowError(
                "converge() requires ConvergenceJobConfig",
                {"workflow": "convergence"},
            )

        settings_to_run = convergence_settings
        if validated_overrides is not None:
            settings_to_run = _apply_convergence_overrides(
                convergence_settings, validated_overrides
            )

        # Deferred import: `execution.py` imports `converge` from this module
        # at module level, so importing `.execution` back here at module
        # level would cycle. By the time converge() actually runs, both
        # modules are already fully loaded.
        from .execution import execute as dispatch_execute

        tracker = MLflowTracker()
        tracker.configure(settings_to_run.tracking)

        multi_run = MultiRunOrchestrator(tracker, dispatch_execute, hooks=hooks)
        orchestrator = ConvergenceOrchestrator(multi_run)

        return orchestrator.execute(settings_to_run)

    return context.run(run_convergence, error_message="Convergence study failed")
