"""Tests for ConvergenceOrchestrator.execute(), confirming multi_run-package-split
behavior is unchanged for convergence's one real caller.

Covers:
- execute() delegates to MultiRunOrchestrator.run_sweep() with failure_policy="fail_fast"
- execute() builds one RunSpec child per (size, repeat) pair, ids "n={n}_r={r}"
- execute() populates ConvergenceResult.mlflow_run_id/mlflow_tracking_uri from
  MultiRunResult.parent_run_id/tracking_uri
- execute() still logs a TOML summary artifact via on_sweep_complete
- execute() propagates a child failure directly (still fail_fast, unchanged behavior)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from dlkit.common.errors import WorkflowError
from dlkit.common.results import (
    ChildOutcome,
    ChildSuccess,
    FailurePolicy,
    MultiRunResult,
    TrainingResult,
    WorkflowResult,
)
from dlkit.engine.tracking.interfaces import IRunContext
from dlkit.engine.workflows.convergence.orchestrator import ConvergenceOrchestrator
from dlkit.engine.workflows.multi_run import MultiRunOrchestrator, RunSpec
from dlkit.infrastructure.config.job_config import ConvergenceJobConfig

PARENT_RUN_ID = "parent-run-id"
TRACKING_URI = "sqlite:///fake.db"


@dataclass
class _FakeMultiRunOrchestrator:
    """Stand-in for MultiRunOrchestrator: fabricates ChildSuccess outcomes.

    Lets ConvergenceOrchestrator.execute() be exercised end-to-end — child
    building, on_sweep_complete-driven summary logging, result unwrapping,
    and tracking-metadata plumbing — without a real MLflow backend.

    Args:
        parent_run: Mock IRunContext passed to on_sweep_complete.
        failing: If True, run_sweep() raises instead of returning, modeling
            fail_fast propagation of a child's exception.
    """

    parent_run: MagicMock
    failing: bool = False
    run_sweep_calls: list[dict[str, Any]] = field(default_factory=list)

    def run_sweep(
        self,
        children: Sequence[RunSpec],
        experiment_name: str,
        parent_run_name: str,
        parent_tags: dict[str, str] | None = None,
        failure_policy: FailurePolicy = "fail_fast",
        on_sweep_complete: Callable[[IRunContext, tuple[ChildOutcome[WorkflowResult], ...]], None]
        | None = None,
    ) -> MultiRunResult[ChildOutcome[WorkflowResult]]:
        self.run_sweep_calls.append(
            {
                "children": list(children),
                "experiment_name": experiment_name,
                "parent_run_name": parent_run_name,
                "parent_tags": parent_tags,
                "failure_policy": failure_policy,
            }
        )
        if self.failing:
            raise WorkflowError("simulated child failure")

        outcomes: tuple[ChildOutcome[WorkflowResult], ...] = tuple(
            ChildSuccess(
                child_id=spec.id,
                label=spec.id,
                run_id=None,
                result=TrainingResult(
                    model_state=None, metrics={}, artifacts={}, duration_seconds=0.1
                ),
            )
            for spec in children
        )
        if on_sweep_complete is not None:
            on_sweep_complete(self.parent_run, outcomes)
        return MultiRunResult(
            parent_run_id=self.parent_run.run_id,
            tracking_uri=TRACKING_URI,
            children=outcomes,
        )


@pytest.fixture
def fake_parent_run() -> MagicMock:
    """Mock IRunContext standing in for the sweep's parent run.

    Returns:
        MagicMock: run_id=PARENT_RUN_ID; log_artifact_content/log_metrics
        are plain mocks for call inspection.
    """
    run = MagicMock()
    run.run_id = PARENT_RUN_ID
    return run


@pytest.fixture
def fake_multi_run(fake_parent_run: MagicMock) -> _FakeMultiRunOrchestrator:
    """A successful FakeMultiRunOrchestrator wired to fake_parent_run.

    Args:
        fake_parent_run: Mock parent run context.

    Returns:
        _FakeMultiRunOrchestrator: failing=False.
    """
    return _FakeMultiRunOrchestrator(parent_run=fake_parent_run)


@pytest.fixture
def failing_multi_run(fake_parent_run: MagicMock) -> _FakeMultiRunOrchestrator:
    """A FakeMultiRunOrchestrator whose run_sweep() raises.

    Args:
        fake_parent_run: Mock parent run context (unused on the failure path).

    Returns:
        _FakeMultiRunOrchestrator: failing=True.
    """
    return _FakeMultiRunOrchestrator(parent_run=fake_parent_run, failing=True)


def test_execute_delegates_with_fail_fast(
    fake_multi_run: _FakeMultiRunOrchestrator,
    convergence_job_settings: ConvergenceJobConfig,
) -> None:
    """execute() always calls run_sweep() with failure_policy="fail_fast".

    Args:
        fake_multi_run: Fake orchestrator recording run_sweep() calls.
        convergence_job_settings: Real ConvergenceJobConfig, trimmed to 2 children.
    """
    orchestrator = ConvergenceOrchestrator(cast(MultiRunOrchestrator, fake_multi_run))
    orchestrator.execute(convergence_job_settings)

    assert fake_multi_run.run_sweep_calls[0]["failure_policy"] == "fail_fast"


def test_execute_builds_one_child_per_size_repeat_pair(
    fake_multi_run: _FakeMultiRunOrchestrator,
    convergence_job_settings: ConvergenceJobConfig,
) -> None:
    """execute() builds ids "n={n}_r={r}" for every (size, repeat) pair.

    Args:
        fake_multi_run: Fake orchestrator recording run_sweep() calls.
        convergence_job_settings: sizes=(10, 20), repeats=1.
    """
    orchestrator = ConvergenceOrchestrator(cast(MultiRunOrchestrator, fake_multi_run))
    orchestrator.execute(convergence_job_settings)

    children = fake_multi_run.run_sweep_calls[0]["children"]
    assert [spec.id for spec in children] == ["n=10_r=0", "n=20_r=0"]


def test_execute_populates_tracking_metadata_from_multi_run_result(
    fake_multi_run: _FakeMultiRunOrchestrator,
    convergence_job_settings: ConvergenceJobConfig,
) -> None:
    """ConvergenceResult.mlflow_run_id/mlflow_tracking_uri come from MultiRunResult.

    Args:
        fake_multi_run: Fake orchestrator returning a known parent run id/URI.
        convergence_job_settings: Real ConvergenceJobConfig, trimmed to 2 children.
    """
    orchestrator = ConvergenceOrchestrator(cast(MultiRunOrchestrator, fake_multi_run))
    result = orchestrator.execute(convergence_job_settings)

    assert result.mlflow_run_id == PARENT_RUN_ID
    assert result.mlflow_tracking_uri == TRACKING_URI


def test_execute_logs_toml_summary_via_on_sweep_complete(
    fake_multi_run: _FakeMultiRunOrchestrator,
    fake_parent_run: MagicMock,
    convergence_job_settings: ConvergenceJobConfig,
) -> None:
    """execute() still logs a convergence_results.toml artifact on the parent run.

    Args:
        fake_multi_run: Fake orchestrator invoking on_sweep_complete internally.
        fake_parent_run: Mock parent run whose log_artifact_content is inspected.
        convergence_job_settings: Real ConvergenceJobConfig, trimmed to 2 children.
    """
    orchestrator = ConvergenceOrchestrator(cast(MultiRunOrchestrator, fake_multi_run))
    orchestrator.execute(convergence_job_settings)

    fake_parent_run.log_artifact_content.assert_called_once()
    _content, artifact_file = fake_parent_run.log_artifact_content.call_args.args
    assert artifact_file == "convergence_results.toml"


def test_execute_propagates_child_failure(
    failing_multi_run: _FakeMultiRunOrchestrator,
    convergence_job_settings: ConvergenceJobConfig,
) -> None:
    """A child failure still propagates directly out of execute() (fail_fast).

    Args:
        failing_multi_run: Fake orchestrator whose run_sweep() raises.
        convergence_job_settings: Real ConvergenceJobConfig, trimmed to 2 children.
    """
    orchestrator = ConvergenceOrchestrator(cast(MultiRunOrchestrator, failing_multi_run))
    with pytest.raises(WorkflowError):
        orchestrator.execute(convergence_job_settings)
