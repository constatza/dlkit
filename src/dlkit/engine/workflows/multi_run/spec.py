"""Public value objects describing a multirun sweep and its children."""

from __future__ import annotations

from dataclasses import dataclass, field

from dlkit.common.results import FailurePolicy
from dlkit.engine.workflows.factories.build_strategy import WorkflowSettings


@dataclass(frozen=True, slots=True, kw_only=True)
class RunSpec:
    """A single child run specification within a multirun sweep.

    Supersedes the training-only ``RunVariant``: a ``RunSpec`` can describe
    any workflow settings type (training, optimization, convergence), so a
    single sweep can mix workflow kinds.

    ``params``/``metadata`` are opaque to DLKit — caller-provided, never
    interpreted here, just threaded through ``expand_child_sources`` so the
    caller can look them up later keyed by ``id``.

    Args:
        id: Stable identifier for this child within the sweep. Used to key
            ``ChildOutcome`` records and must be unique across a sweep.
        label: Human-readable label for this child (e.g. for logging/UI).
        settings: Fully-patched workflow settings for this child.
        run_name: MLflow child run name, e.g. ``"n=100_r=0"``.
        tags: Optional MLflow tags for this child's run.
        params: Opaque caller-supplied parameters, keyed by ``id`` by the
            caller — not interpreted by DLKit.
        metadata: Opaque caller-supplied metadata, keyed by ``id`` by the
            caller — not interpreted by DLKit.
    """

    id: str
    label: str
    settings: WorkflowSettings
    run_name: str
    tags: dict[str, str] = field(default_factory=dict)
    params: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class MultiRunSpec:
    """Full specification of a multirun sweep: parent run identity plus children.

    Args:
        experiment_name: MLflow experiment name shared by the parent and
            every child run.
        parent_run_name: MLflow name for the parent sweep run.
        parent_tags: Optional tags for the parent run.
        failure_policy: How the sweep reacts when a child run fails.
        children: Ordered child run specifications.
    """

    experiment_name: str
    parent_run_name: str
    parent_tags: dict[str, str] = field(default_factory=dict)
    failure_policy: FailurePolicy = "fail_fast"
    children: tuple[RunSpec, ...]
