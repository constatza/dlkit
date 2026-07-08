"""Runtime-owned convergence workflow entrypoint."""

from __future__ import annotations

from dlkit.common.errors import WorkflowError
from dlkit.common.hooks import LifecycleHooks
from dlkit.common.results import ConvergenceResult
from dlkit.engine.tracking.mlflow_tracker import MLflowTracker
from dlkit.engine.training.vanilla_executor import VanillaExecutor
from dlkit.engine.workflows.convergence.orchestrator import ConvergenceOrchestrator
from dlkit.engine.workflows.factories.build_factory import BuildFactory
from dlkit.engine.workflows.multi_run import MultiRunOrchestrator
from dlkit.infrastructure.config.job_config import ConvergenceJobConfig
from dlkit.infrastructure.utils.error_handling import raise_error
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

    return settings.patch(patch) if patch else settings  # type: ignore[return-value]


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

    try:
        convergence_settings = context.settings
        if not isinstance(convergence_settings, ConvergenceJobConfig):
            raise WorkflowError(
                "converge() requires ConvergenceJobConfig",
                {"workflow": "convergence"},
            )

        if validated_overrides is not None:
            convergence_settings = _apply_convergence_overrides(
                convergence_settings, validated_overrides
            )

        tracker = MLflowTracker()
        tracker.configure(convergence_settings.tracking)

        build_factory = BuildFactory()
        executor = VanillaExecutor()
        multi_run = MultiRunOrchestrator(build_factory, executor, tracker, hooks=hooks)
        orchestrator = ConvergenceOrchestrator(multi_run)

        return context.run_with_path_context(lambda: orchestrator.execute(convergence_settings))

    except Exception as exc:
        if isinstance(exc, WorkflowError):
            raise
        raise_error("Convergence study failed", exc)
