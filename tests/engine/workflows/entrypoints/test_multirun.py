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

from dlkit.common.results import ChildSuccess, MultiRunResult, OptimizationResult, TrainingResult
from dlkit.engine.tracking.run_queries import find_child_run_ids
from dlkit.engine.workflows.entrypoints.multirun import run_multirun, run_multirun_spec
from dlkit.engine.workflows.multi_run import MultiRunSpec, RunSpec
from dlkit.infrastructure.config.job_config import MultiRunJobConfig, TrainingJobConfig


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
