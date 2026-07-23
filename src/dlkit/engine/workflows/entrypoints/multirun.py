"""Runtime-owned multirun sweep entrypoint.

Converts a validated :class:`~dlkit.infrastructure.config.job_config.MultiRunJobConfig`
(or an already-built :class:`~dlkit.engine.workflows.multi_run.MultiRunSpec`) into the
engine's generic :class:`~dlkit.engine.workflows.multi_run.MultiRunOrchestrator`, wiring
up its ``dispatch`` collaborator with the same deferred-import pattern
:mod:`convergence` uses, for the same reason: ``entrypoints/execution.py`` imports
``converge`` (and, transitively, this package's siblings) at module level, so a
module-level ``from .execution import execute`` here would risk the same import cycle.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from dlkit.common.errors import ConfigValidationError, WorkflowError
from dlkit.common.hooks import LifecycleHooks
from dlkit.common.results import ChildOutcome, FailurePolicy, MultiRunResult, WorkflowResult
from dlkit.engine.tracking.mlflow_tracker import MLflowTracker
from dlkit.engine.workflows.factories.tracking_flag import apply_mlflow_flag
from dlkit.engine.workflows.multi_run import (
    ChildSource,
    ExplicitFileSource,
    GlobSource,
    MultiRunOrchestrator,
    MultiRunSpec,
    RunSpec,
    expand_child_sources,
)
from dlkit.infrastructure.config.job_config import ChildEntryConfig, MultiRunJobConfig
from dlkit.infrastructure.utils.error_handling import raise_error
from dlkit.infrastructure.utils.logging_config import get_logger

logger = get_logger(__name__)

# Glob shorthand: a `files` string containing any of these is treated as a
# GlobSource pattern rather than a single explicit file path.
_GLOB_CHARS = frozenset("*?[")


def _is_glob_pattern(value: str) -> bool:
    """Return True if value contains a glob metacharacter (``*``, ``?``, ``[``).

    Args:
        value: Candidate ``files`` string.

    Returns:
        True if value should be treated as a glob pattern.
    """
    return any(char in value for char in _GLOB_CHARS)


def _child_source_from_entry(entry: ChildEntryConfig) -> ChildSource:
    """Convert one validated ``ChildEntryConfig`` into an engine-layer ``ChildSource``.

    ``entry.files`` has already been resolved to absolute path(s)/pattern by
    ``load_job()`` — this function only decides glob-vs-explicit and shapes
    the result into the matching ``ChildSource`` variant.

    Args:
        entry: Validated child entry.

    Returns:
        ``GlobSource`` for a single glob-pattern string; ``ExplicitFileSource``
        for a single file or a merge list otherwise.

    Raises:
        ConfigValidationError: A list of files contains a glob-pattern string
            — glob shorthand must be a single string, not a list.
    """
    label = entry.label or entry.id
    match entry.files:
        case str() as pattern if _is_glob_pattern(pattern):
            pattern_path = Path(pattern)
            return GlobSource(
                id_prefix=f"{entry.id}_",
                pattern=pattern_path.name,
                base_dir=pattern_path.parent,
                run_type=entry.run_type,
                tags=entry.tags,
                params=entry.params,
                metadata=entry.metadata,
            )
        case str() as single_file:
            return ExplicitFileSource(
                id=entry.id,
                label=label,
                files=(Path(single_file),),
                run_type=entry.run_type,
                tags=entry.tags,
                params=entry.params,
                metadata=entry.metadata,
            )
        case list() as files_list:
            if any(_is_glob_pattern(f) for f in files_list if isinstance(f, str)):
                raise ConfigValidationError(
                    f"Child {entry.id!r}: glob patterns must be a single string, "
                    f"not a list: {files_list!r}"
                )
            return ExplicitFileSource(
                id=entry.id,
                label=label,
                files=tuple(Path(f) for f in files_list),
                run_type=entry.run_type,
                tags=entry.tags,
                params=entry.params,
                metadata=entry.metadata,
            )
        case _:
            raise ConfigValidationError(
                f"Child {entry.id!r}: files must be a string or list of strings, "
                f"got {type(entry.files).__name__}"
            )


def build_child_sources(entries: Sequence[ChildEntryConfig]) -> tuple[ChildSource, ...]:
    """Convert every ``[[multirun.runs]]`` entry into a ``ChildSource``.

    Pure conversion step, kept separate from expansion so CLI ``validate``
    and ``run_multirun()`` can share it without either triggering the other's
    side effects.

    Args:
        entries: Validated child entries from a ``MultiRunJobConfig``.

    Returns:
        One ``ChildSource`` per entry, in order.
    """
    return tuple(_child_source_from_entry(entry) for entry in entries)


def _run_sweep(
    *,
    children: Sequence[RunSpec],
    experiment_name: str,
    parent_run_name: str,
    parent_tags: dict[str, str] | None,
    failure_policy: FailurePolicy,
    hooks: LifecycleHooks | None,
) -> MultiRunResult[ChildOutcome[WorkflowResult]]:
    """Shared sweep execution: build a tracker/orchestrator and run children.

    The parent run's tracker is configured from the first child's own
    settings (mlflow-flag-forced), matching ``MultiRunOrchestrator._run_one``'s
    existing per-child behavior — a multirun sweep's entire purpose is
    parent/child MLflow linkage, so tracking is always enabled here rather
    than silently producing an untrackable sweep.

    Args:
        children: Expanded child run specifications.
        experiment_name: MLflow experiment name shared by the parent and
            every child run.
        parent_run_name: MLflow name for the parent sweep run.
        parent_tags: Optional tags for the parent run.
        failure_policy: How the sweep reacts when a child run fails.
        hooks: Optional lifecycle hooks.

    Returns:
        MultiRunResult with the parent run id, tracking URI, and one
        ChildOutcome per child, in expansion order.

    Raises:
        WorkflowError: If the sweep fails for any reason (empty sweep, child
            dispatch failure under ``fail_fast``, tracker configuration
            failure).
    """
    if not children:
        raise ConfigValidationError("Multirun sweep requires at least one child run.")

    try:
        # Deferred import: `execution.py` imports `converge` from this
        # package at module level, so importing `.execution` back here at
        # module level would risk the same cycle `convergence.py` hit.
        from .execution import execute as dispatch_execute

        first_child_settings = apply_mlflow_flag(children[0].settings, mlflow=True)
        tracking = getattr(first_child_settings, "tracking", None)
        if tracking is None:
            raise ConfigValidationError(
                f"Multirun child {children[0].id!r} settings has no `tracking` "
                "section; cannot configure the parent run's tracker."
            )

        tracker = MLflowTracker()
        tracker.configure(tracking)

        orchestrator = MultiRunOrchestrator(tracker, dispatch_execute, hooks=hooks)
        return orchestrator.run_sweep(
            children=children,
            experiment_name=experiment_name,
            parent_run_name=parent_run_name,
            parent_tags=parent_tags or {},
            failure_policy=failure_policy,
        )
    except Exception as exc:
        if isinstance(exc, WorkflowError):
            raise
        raise_error("Multirun sweep failed", exc)


def run_multirun(
    settings: MultiRunJobConfig,
    *,
    hooks: LifecycleHooks | None = None,
) -> MultiRunResult[ChildOutcome[WorkflowResult]]:
    """Run a multirun sweep from a validated ``MultiRunJobConfig``.

    Converts ``settings.runs`` into ``ChildSource``s (explicit file merges or
    glob shorthand), expands them into ``RunSpec``s, and executes the sweep
    through ``MultiRunOrchestrator``.

    Args:
        settings: Validated multirun job configuration.
        hooks: Optional lifecycle hooks fired around the parent run and
            forwarded into every child's own workflow execution.

    Returns:
        MultiRunResult with the parent run id, tracking URI, and one
        ChildOutcome per child, in expansion order.

    Raises:
        WorkflowError: If child expansion or sweep execution fails (duplicate
            ids, zero glob matches, nested multirun children, empty sweep,
            or a child dispatch failure under ``fail_fast``).
    """
    logger.info(
        "Multirun | experiment={} parent_run_name={} children={}",
        settings.experiment_name,
        settings.parent_run_name,
        len(settings.runs),
    )
    sources = build_child_sources(settings.runs)
    try:
        children = expand_child_sources(sources)
    except ConfigValidationError:
        raise
    except Exception as exc:
        raise_error("Multirun child expansion failed", exc)

    return _run_sweep(
        children=children,
        experiment_name=settings.experiment_name,
        parent_run_name=settings.parent_run_name,
        parent_tags=None,
        failure_policy=settings.failure_policy,
        hooks=hooks,
    )


def run_multirun_spec(
    spec: MultiRunSpec,
    *,
    hooks: LifecycleHooks | None = None,
) -> MultiRunResult[ChildOutcome[WorkflowResult]]:
    """Run a multirun sweep from an already-built ``MultiRunSpec``.

    For callers who construct children programmatically (e.g. from
    already-loaded settings via ``LoadedSettingsSource``) rather than loading
    a ``MultiRunJobConfig`` from TOML — no config parsing/expansion needed,
    ``spec.children`` is already a ``tuple[RunSpec, ...]``.

    Args:
        spec: Fully-specified sweep: parent identity plus expanded children.
        hooks: Optional lifecycle hooks fired around the parent run and
            forwarded into every child's own workflow execution.

    Returns:
        MultiRunResult with the parent run id, tracking URI, and one
        ChildOutcome per child, in expansion order.

    Raises:
        WorkflowError: If the sweep fails for any reason (empty sweep, or a
            child dispatch failure under ``fail_fast``).
    """
    logger.info(
        "Multirun (spec) | experiment={} parent_run_name={} children={}",
        spec.experiment_name,
        spec.parent_run_name,
        len(spec.children),
    )
    return _run_sweep(
        children=spec.children,
        experiment_name=spec.experiment_name,
        parent_run_name=spec.parent_run_name,
        parent_tags=spec.parent_tags,
        failure_policy=spec.failure_policy,
        hooks=hooks,
    )
