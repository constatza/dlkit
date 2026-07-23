"""Real-MLflow tests for the multirun runtime entrypoints.

Covers `run_multirun()` (config-driven, TOML files on disk) and
`run_multirun_spec()` (already-built settings, no config parsing), against a
real sqlite-backed MLflow tracking store under `tmp_path` — mirroring the
real-tracking-store conventions in
`tests/engine/workflows/multi_run/test_orchestrator.py` (mocked dispatch) and
`tests/integration/test_evaluate_multirun_integration.py` (real dispatch).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlkit.common.errors import ConfigValidationError
from dlkit.common.hooks import ChildPlannedEvent, LifecycleHooks, RunCreatedEvent
from dlkit.common.results import (
    ChildSuccess,
    MultiRunResult,
    OptimizationResult,
    TrainingResult,
    WorkflowResult,
)
from dlkit.engine.tracking.run_queries import find_child_run_ids
from dlkit.engine.workflows.entrypoints.multirun import (
    build_child_sources,
    run_multirun,
    run_multirun_spec,
)
from dlkit.engine.workflows.multi_run import (
    ExplicitFileSource,
    MultiRunSpec,
    RunSpec,
    expand_child_sources,
)
from dlkit.infrastructure.config.job_config import (
    ChildEntryConfig,
    MultiRunJobConfig,
    TrainingJobConfig,
)


def _multirun_config(
    *,
    experiment_name: str,
    parent_run_name: str,
    runs: list[dict[str, object]],
) -> MultiRunJobConfig:
    """Build a MultiRunJobConfig directly (flat fields — no TOML hoisting needed).

    Args:
        experiment_name: MLflow experiment name for the sweep.
        parent_run_name: MLflow name for the parent sweep run.
        runs: Raw `[[multirun.runs]]`-shaped entry dicts.

    Returns:
        Validated MultiRunJobConfig.
    """
    return MultiRunJobConfig.model_validate(
        {
            "run": {"type": "multirun"},
            "experiment_name": experiment_name,
            "parent_run_name": parent_run_name,
            "runs": runs,
        }
    )


def test_run_multirun_config_train_only_sweep_round_trips(
    train_child_paths: tuple[Path, Path],
) -> None:
    """A homogeneous train-only sweep round-trips through config to a real MultiRunResult."""
    child_a, child_b = train_child_paths
    settings = _multirun_config(
        experiment_name="sweep-train-only",
        parent_run_name="sweep-parent-train-only",
        runs=[
            {"id": "child-a", "label": "Child A", "files": [str(child_a)]},
            {"id": "child-b", "label": "Child B", "files": [str(child_b)]},
        ],
    )

    result = run_multirun(settings)

    assert isinstance(result, MultiRunResult)
    assert len(result.children) == 2
    for outcome in result.children:
        assert isinstance(outcome, ChildSuccess)
        assert isinstance(outcome.result, TrainingResult)

    child_run_ids = find_child_run_ids(
        parent_run_id=result.parent_run_id, tracking_uri=result.tracking_uri
    )
    assert set(child_run_ids) == {outcome.run_id for outcome in result.children}


def test_run_multirun_config_heterogeneous_sweep_train_and_search(
    train_and_search_child_paths: tuple[Path, Path],
) -> None:
    """A train child and a search child both succeed, tagged under the same parent."""
    train_path, search_path = train_and_search_child_paths
    settings = _multirun_config(
        experiment_name="sweep-hetero",
        parent_run_name="sweep-parent-hetero",
        runs=[
            {"id": "train-child", "label": "Train Child", "files": [str(train_path)]},
            {"id": "search-child", "label": "Search Child", "files": [str(search_path)]},
        ],
    )

    result = run_multirun(settings)

    assert len(result.children) == 2
    train_outcome, search_outcome = result.children
    assert isinstance(train_outcome, ChildSuccess)
    assert isinstance(train_outcome.result, TrainingResult)
    assert isinstance(search_outcome, ChildSuccess)
    assert isinstance(search_outcome.result, OptimizationResult)

    # Assert both children's own MLflow runs are actually tagged with the
    # sweep's parent run id, against the real tracking store (not just the
    # in-memory result). OptimizationResult now carries mlflow_run_id, so the
    # search child is taggable the same as the train child — previously a
    # known gap (OptimizationResult had no run id to extract).
    assert search_outcome.run_id is not None
    child_run_ids = find_child_run_ids(
        parent_run_id=result.parent_run_id, tracking_uri=result.tracking_uri
    )
    assert train_outcome.run_id in child_run_ids
    assert search_outcome.run_id in child_run_ids


def test_run_multirun_spec_train_only_sweep(
    train_job_configs: tuple[TrainingJobConfig, TrainingJobConfig],
) -> None:
    """run_multirun_spec() executes an already-built sweep with no config parsing."""
    cfg_a, cfg_b = train_job_configs
    children = (
        RunSpec(id="a", label="a", settings=cfg_a, run_name="variant-a"),
        RunSpec(id="b", label="b", settings=cfg_b, run_name="variant-b"),
    )
    spec = MultiRunSpec(
        experiment_name="spec-sweep",
        parent_run_name="spec-sweep-parent",
        children=children,
    )

    result = run_multirun_spec(spec)

    assert isinstance(result, MultiRunResult)
    assert len(result.children) == 2
    for outcome in result.children:
        assert isinstance(outcome, ChildSuccess)
        assert isinstance(outcome.result, TrainingResult)

    child_run_ids = find_child_run_ids(
        parent_run_id=result.parent_run_id, tracking_uri=result.tracking_uri
    )
    assert set(child_run_ids) == {outcome.run_id for outcome in result.children}


# ---------------------------------------------------------------------------
# Tests: TOML `patches` / `parent_tags` config-loading consistency
# ---------------------------------------------------------------------------


def test_build_child_sources_threads_patches_into_explicit_file_source(
    train_child_paths: tuple[Path, Path],
) -> None:
    """A `[[multirun.runs]]` entry's `patches` reaches `ExplicitFileSource.patches`.

    Regression guard for the TOML schema gap: `ExplicitFileSource.patches`
    already existed at the engine layer, but `ChildEntryConfig` had no field
    to carry it from TOML.
    """
    child_a, _child_b = train_child_paths
    entry = ChildEntryConfig(
        id="child-a", label="Child A", files=str(child_a), patches={"run.seed": 99}
    )

    (source,) = build_child_sources([entry])

    assert isinstance(source, ExplicitFileSource)
    assert source.patches == {"run.seed": 99}


def test_build_child_sources_rejects_patches_on_glob_entry() -> None:
    """`patches` on a glob-sourced entry raises — one dict can't fit every match."""
    entry = ChildEntryConfig(id="variants", files="variants/*.toml", patches={"run.seed": 99})

    with pytest.raises(ConfigValidationError, match="glob-sourced entry"):
        build_child_sources([entry])


def test_expand_child_sources_applies_patches_to_child_settings(
    train_child_paths: tuple[Path, Path],
) -> None:
    """A patched `ExplicitFileSource` produces a `RunSpec` whose settings carry the patch."""
    child_a, _child_b = train_child_paths
    source = ExplicitFileSource(
        id="child-a",
        label="Child A",
        files=(child_a,),
        patches={"run.seed": 99},
    )

    (spec,) = expand_child_sources([source])

    assert spec.settings.run.seed == 99


def test_run_multirun_config_applies_parent_tags_to_parent_run(
    train_child_paths: tuple[Path, Path],
) -> None:
    """`MultiRunJobConfig.parent_tags` reaches the real parent MLflow run's tags."""
    from dlkit.engine.tracking.mlflow_client_factory import MLflowClientFactory

    child_a, child_b = train_child_paths
    settings = MultiRunJobConfig.model_validate(
        {
            "run": {"type": "multirun"},
            "experiment_name": "sweep-parent-tags",
            "parent_run_name": "sweep-parent-tags-parent",
            "parent_tags": {"team": "platform"},
            "runs": [
                {"id": "child-a", "label": "Child A", "files": [str(child_a)]},
                {"id": "child-b", "label": "Child B", "files": [str(child_b)]},
            ],
        }
    )

    result = run_multirun(settings)

    client = MLflowClientFactory.create_client(result.tracking_uri)
    parent_run = client.get_run(result.parent_run_id)
    assert parent_run.data.tags.get("team") == "platform"


# ---------------------------------------------------------------------------
# Tests: hooks reachable from the public entrypoints
# ---------------------------------------------------------------------------


def test_run_multirun_forwards_hooks_to_children(
    train_child_paths: tuple[Path, Path],
) -> None:
    """`run_multirun(settings, hooks=...)` fires on_run_created for real child runs.

    Regression guard for the public multirun API's hooks gap: hooks were
    already threaded through the engine-layer entrypoint, but nothing
    exercised it end to end against real child dispatch.
    """
    child_a, child_b = train_child_paths
    settings = _multirun_config(
        experiment_name="sweep-hooks",
        parent_run_name="sweep-hooks-parent",
        runs=[
            {"id": "child-a", "label": "Child A", "files": [str(child_a)]},
            {"id": "child-b", "label": "Child B", "files": [str(child_b)]},
        ],
    )
    recorded: list[RunCreatedEvent] = []
    hooks = LifecycleHooks(on_run_created=recorded.append)

    result = run_multirun(settings, hooks=hooks)

    assert isinstance(result, MultiRunResult)
    kinds = [event.kind for event in recorded]
    assert kinds.count("sweep") == 1
    assert kinds.count("train") == 2


def test_run_multirun_forwards_child_planned_and_completed_hooks(
    train_child_paths: tuple[Path, Path],
) -> None:
    """`run_multirun(settings, hooks=...)` fires on_child_planned/on_child_completed per child.

    Proves the whole hooks chain works end to end through the real public
    entrypoint (`run_multirun` -> `MultiRunOrchestrator._run_one`), not just
    the orchestrator unit tests with a mocked dispatcher.
    """
    child_a, child_b = train_child_paths
    settings = _multirun_config(
        experiment_name="sweep-child-hooks",
        parent_run_name="sweep-child-hooks-parent",
        runs=[
            {"id": "child-a", "label": "Child A", "files": [str(child_a)]},
            {"id": "child-b", "label": "Child B", "files": [str(child_b)]},
        ],
    )
    recorded_planned: list[ChildPlannedEvent] = []
    recorded_completed: list[ChildSuccess[WorkflowResult]] = []
    hooks = LifecycleHooks(
        on_child_planned=recorded_planned.append,
        on_child_completed=recorded_completed.append,
    )

    result = run_multirun(settings, hooks=hooks)

    assert isinstance(result, MultiRunResult)
    assert len(recorded_planned) == 2
    assert [event.child_id for event in recorded_planned] == ["child-a", "child-b"]

    assert len(recorded_completed) == 2
    assert [success.child_id for success in recorded_completed] == ["child-a", "child-b"]
    assert all(isinstance(success, ChildSuccess) for success in recorded_completed)
