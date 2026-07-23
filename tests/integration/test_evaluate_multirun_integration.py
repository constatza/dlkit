"""End-to-end test: train a real multirun sweep, then batch-evaluate its children.

Verifies ``evaluate_multirun()`` (``dlkit.interfaces.api.functions.core``)
composes an ``ExistingRunsSource`` + the general ``MultiRunOrchestrator``
pipeline to fan a single ``evaluate()`` call out over every child run of a
real ``MultiRunOrchestrator`` sweep, using genuine checkpoints/MLflow runs
(not mocked), plus a focused unit-level test that it fails loudly when the
parent run has no children.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from dlkit.common.errors import WorkflowError
from dlkit.common.results import ChildFailure, ChildSuccess, EvaluationResult, MultiRunResult
from dlkit.engine.tracking.mlflow_tracker import MLflowTracker
from dlkit.engine.tracking.run_queries import find_child_run_ids
from dlkit.engine.workflows.entrypoints import execute
from dlkit.engine.workflows.multi_run import MultiRunOrchestrator, RunSpec
from dlkit.infrastructure.config.job_config import InferenceJobConfig, TrainingJobConfig
from dlkit.infrastructure.config.tracking_settings import TrackingSettings
from dlkit.interfaces.api.functions.core import evaluate_multirun

# Same model/data shape as tests/integration/conftest.py's _make_training_job_config,
# since InferenceJobConfig must describe the identical model + data the checkpoints
# were trained with (component "name"/"class" is write-only in the settings models, so
# it can't be recovered by dumping a validated TrainingJobConfig back out).
FEATURE_SIZE = 4
NUM_VARIANTS = 3


def _build_inference_settings(minimal_dataset: dict[str, Path]) -> InferenceJobConfig:
    """Build an InferenceJobConfig matching the FFNN trained by the sweep variants.

    ``model.checkpoint`` is required by ``InferenceJobConfig`` but unused here: the
    per-child checkpoint resolved via ``ExistingRunsSource`` always overrides
    ``settings.model.checkpoint`` before dispatch.
    """
    return InferenceJobConfig.model_validate(
        {
            "run": {"type": "predict"},
            "experiment": {"name": "integration_test"},
            "model": {
                "class": "FFNN",
                "module_path": "dlkit.domain.nn",
                "hidden_size": FEATURE_SIZE,
                "num_layers": 0,
                "checkpoint": "unused-placeholder.ckpt",
            },
            "data": {
                "class": "FlexibleDataset",
                "module_path": "dlkit.engine.data.datasets",
                "batch_size": 4,
                "num_workers": 0,
                "shuffle": False,
                "pin_memory": False,
                "persistent_workers": False,
                "features": [
                    {"name": "x", "path": str(minimal_dataset["features"]), "format": "npy"}
                ],
                "targets": [
                    {"name": "y", "path": str(minimal_dataset["targets"]), "format": "npy"}
                ],
            },
        }
    )


def _split_filepath(training_settings: TrainingJobConfig) -> Path:
    """The single `splits/*.json` file training persisted for this run.

    A run-checkpoint-resolved checkpoint is downloaded to an arbitrary temp
    directory, not colocated next to a `splits/` directory the way a local
    training-output checkpoint is — so `evaluate()`'s default colocated-split
    auto-location (`DatasetBuilder._resolve_colocated_split_filepath`) can't
    apply there. Real callers evaluating a run-based checkpoint must set
    `data.splits.filepath` explicitly; these tests do the same.
    """
    training_cfg = training_settings.training
    assert training_cfg is not None
    trainer_cfg = training_cfg.trainer
    assert trainer_cfg is not None
    split_files = list(Path(trainer_cfg.default_root_dir).glob("splits/*.json"))
    assert len(split_files) == 1, f"expected exactly one split file, found {split_files}"
    return split_files[0]


def _with_split_filepath(settings: InferenceJobConfig, split_filepath: Path) -> InferenceJobConfig:
    """Point `settings.data.splits.filepath` at an explicit split file."""
    assert settings.data is not None
    return settings.model_copy(
        update={
            "data": settings.data.model_copy(
                update={
                    "splits": settings.data.splits.model_copy(update={"filepath": split_filepath})
                }
            )
        }
    )


@pytest.fixture
def sweep_variant_settings(
    training_settings: TrainingJobConfig, tmp_path: Path
) -> tuple[TrainingJobConfig, ...]:
    """NUM_VARIANTS training configs sharing data/seed, each with an isolated
    checkpoint output dir and a shared real (sqlite) MLflow tracking backend
    — mirrors `test_evaluate_integration.py`'s `training_settings_with_checkpoint_and_mlflow`
    fixture, replicated per variant since each needs its own `default_root_dir` to avoid
    checkpoint/split-file collisions.
    """
    training_cfg = training_settings.training
    assert training_cfg is not None
    trainer_cfg = training_cfg.trainer
    assert trainer_cfg is not None

    mlruns_dir = tmp_path / "sweep_mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    tracking = TrackingSettings(
        backend="mlflow", uri=f"sqlite:///{(mlruns_dir / 'mlflow.db').as_posix()}"
    )

    def _variant(index: int) -> TrainingJobConfig:
        root = tmp_path / f"sweep_output_{index}"
        root.mkdir(parents=True, exist_ok=True)
        return training_settings.model_copy(
            update={
                "training": training_cfg.model_copy(
                    update={
                        "trainer": trainer_cfg.model_copy(
                            update={
                                "enable_checkpointing": True,
                                "default_root_dir": root,
                                "fast_dev_run": False,
                            }
                        )
                    }
                ),
                "tracking": tracking,
            }
        )

    return tuple(_variant(i) for i in range(NUM_VARIANTS))


@pytest.fixture
def trained_sweep(
    sweep_variant_settings: tuple[TrainingJobConfig, ...],
) -> tuple[str, tuple[TrainingJobConfig, ...]]:
    """Train `sweep_variant_settings` via a real MultiRunOrchestrator.

    Returns:
        Tuple of (parent_run_id, sweep_variant_settings) — the parent sweep run id
        and the per-variant settings used to derive the eval-side split file.
    """
    tracker = MLflowTracker()
    tracker.configure(sweep_variant_settings[0].tracking)
    orchestrator = MultiRunOrchestrator(tracker, execute)

    children = [
        RunSpec(id=f"variant-{i}", label=f"variant-{i}", settings=settings, run_name=f"variant-{i}")
        for i, settings in enumerate(sweep_variant_settings)
    ]

    result = orchestrator.run_sweep(
        children=children,
        experiment_name="sweep_experiment",
        parent_run_name="sweep_parent",
    )

    return result.parent_run_id, sweep_variant_settings


@pytest.fixture
def multirun_inference_settings(
    minimal_dataset: dict[str, Path],
    trained_sweep: tuple[str, tuple[TrainingJobConfig, ...]],
) -> InferenceJobConfig:
    """InferenceJobConfig matching the sweep's model/data, pointed at its split file."""
    _parent_run_id, variant_settings = trained_sweep
    settings = _build_inference_settings(minimal_dataset).model_copy(
        update={"tracking": variant_settings[0].tracking}
    )
    return _with_split_filepath(settings, _split_filepath(variant_settings[0]))


def test_evaluate_multirun_evaluates_every_child_run(
    trained_sweep: tuple[str, tuple[TrainingJobConfig, ...]],
    multirun_inference_settings: InferenceJobConfig,
) -> None:
    parent_run_id, variant_settings = trained_sweep
    expected_run_ids = set(
        find_child_run_ids(
            parent_run_id=parent_run_id, tracking_uri=variant_settings[0].tracking.uri
        )
    )
    assert len(expected_run_ids) == NUM_VARIANTS

    result = evaluate_multirun(multirun_inference_settings, parent_run_id=parent_run_id)

    try:
        assert isinstance(result, MultiRunResult)
        # The evaluate sweep opens its own new parent run, distinct from the
        # training sweep's parent it evaluated.
        assert result.parent_run_id != parent_run_id
        assert len(result.children) == NUM_VARIANTS
        assert all(isinstance(child, ChildSuccess) for child in result.children)
        # child_id carries the source checkpoint's run identity.
        assert {child.child_id for child in result.children} == expected_run_ids
        for child in result.children:
            assert isinstance(child.result, EvaluationResult)
            assert child.result.metrics.keys() == {"mae", "rmse", "r2"}
            assert all(isinstance(v, float) for v in child.result.metrics.values())
            assert child.result.predictions.shape[0] == child.result.targets.shape[0] > 0
    finally:
        for child in result.children:
            if isinstance(child, ChildSuccess):
                for fig in child.result.figures.values():
                    plt.close(fig)


def test_evaluate_multirun_tags_children_and_logs_each_child(
    trained_sweep: tuple[str, tuple[TrainingJobConfig, ...]],
    multirun_inference_settings: InferenceJobConfig,
) -> None:
    """Every child is always logged to MLflow now (the orchestrator forces
    tracking on for every dispatched child), and is a distinct run from its
    own source checkpoint's run — `child_id` (source) and `run_id` (this
    child's own new evaluate-logging run) must never collide.
    """
    parent_run_id, variant_settings = trained_sweep

    result = evaluate_multirun(multirun_inference_settings, parent_run_id=parent_run_id)

    try:
        assert len(result.children) == NUM_VARIANTS
        for child in result.children:
            assert isinstance(child, ChildSuccess)
            assert child.run_id
            assert child.result.mlflow_run_id == child.run_id
            assert child.child_id != child.run_id
        tagged_children = find_child_run_ids(
            parent_run_id=result.parent_run_id, tracking_uri=variant_settings[0].tracking.uri
        )
        assert set(tagged_children) == {child.run_id for child in result.children}
    finally:
        for child in result.children:
            if isinstance(child, ChildSuccess):
                for fig in child.result.figures.values():
                    plt.close(fig)


def test_evaluate_multirun_raises_when_parent_has_no_children(
    monkeypatch: pytest.MonkeyPatch,
    minimal_dataset: dict[str, Path],
) -> None:
    """A parent run with zero active children is a caller mistake, not an empty batch.

    Covered via monkeypatching `find_child_run_ids` (rather than a full real-MLflow
    "parent with zero children" fixture) since `find_child_run_ids` itself already has
    dedicated real-MLflow coverage for this exact case in
    `tests/engine/tracking/test_run_queries.py`.
    """

    def _raise_no_children(
        *, parent_run_id: str, tracking_uri: str | None = None
    ) -> tuple[str, ...]:
        raise WorkflowError(
            f"Parent run {parent_run_id!r} has no active child runs.",
            {"parent_run_id": parent_run_id},
        )

    monkeypatch.setattr(
        "dlkit.engine.workflows.multi_run.child_source.find_child_run_ids", _raise_no_children
    )

    settings = _build_inference_settings(minimal_dataset)

    with pytest.raises(WorkflowError, match="no active child runs"):
        evaluate_multirun(settings, parent_run_id="lonely-parent")


def test_evaluate_multirun_callable_settings_resolves_per_child(
    trained_sweep: tuple[str, tuple[TrainingJobConfig, ...]],
    minimal_dataset: dict[str, Path],
) -> None:
    """A per-child settings resolver lets each child use its own settings.

    Regression guard for gap 3 (`docs/general-multirun-api-requirements.md`):
    before this, `evaluate_multirun()` could only vary `model.checkpoint`
    per child, sharing one `data.splits.filepath` (and everything else)
    across the whole sweep — structurally unable to give each child its own
    dataset. This dispatches real per-child evaluate() calls, each pointed
    at that child's own split file via the resolver, proving the callable
    path drives real dispatch correctly rather than just accepting the type.
    """
    parent_run_id, variant_settings = trained_sweep
    tracking_uri = variant_settings[0].tracking.uri
    run_ids = find_child_run_ids(parent_run_id=parent_run_id, tracking_uri=tracking_uri)
    assert len(run_ids) == NUM_VARIANTS

    per_child_settings = {
        run_id: _with_split_filepath(
            _build_inference_settings(minimal_dataset).model_copy(
                update={"tracking": variant.tracking}
            ),
            _split_filepath(variant),
        )
        for run_id, variant in zip(run_ids, variant_settings, strict=True)
    }

    result = evaluate_multirun(
        lambda run_id: per_child_settings[run_id],
        parent_run_id=parent_run_id,
        tracking_uri=tracking_uri,
    )

    try:
        assert isinstance(result, MultiRunResult)
        assert len(result.children) == NUM_VARIANTS
        assert all(isinstance(child, ChildSuccess) for child in result.children)
        assert {child.child_id for child in result.children} == set(run_ids)
    finally:
        for child in result.children:
            if isinstance(child, ChildSuccess):
                for fig in child.result.figures.values():
                    plt.close(fig)


def test_evaluate_multirun_continue_policy_records_child_failures(
    trained_sweep: tuple[str, tuple[TrainingJobConfig, ...]],
    multirun_inference_settings: InferenceJobConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`failure_policy="continue"` records a `ChildFailure` per bad child
    instead of aborting the whole batch — a capability the old bespoke
    `evaluate_multirun()` never had (it was strictly all-or-nothing).
    """
    parent_run_id, _variant_settings = trained_sweep

    # `dlkit.engine.workflows.entrypoints`'s __init__ re-exports the `evaluate`
    # *function*, shadowing the `evaluate` submodule on the package object
    # (same shadowing documented for `dlkit.interfaces.inference.evaluate`) —
    # so `import_module` looks the submodule up in `sys.modules` directly.
    evaluate_module = import_module("dlkit.engine.workflows.entrypoints.evaluate")

    original = evaluate_module.download_checkpoint_artifact
    calls = {"count": 0}

    def _fail_first_then_delegate(*args: object, **kwargs: object) -> Path:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("simulated checkpoint download failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(evaluate_module, "download_checkpoint_artifact", _fail_first_then_delegate)

    result = evaluate_multirun(
        multirun_inference_settings, parent_run_id=parent_run_id, failure_policy="continue"
    )

    try:
        assert len(result.children) == NUM_VARIANTS
        failures = [child for child in result.children if isinstance(child, ChildFailure)]
        successes = [child for child in result.children if isinstance(child, ChildSuccess)]
        assert len(failures) == 1
        assert len(successes) == NUM_VARIANTS - 1
    finally:
        for child in result.children:
            if isinstance(child, ChildSuccess):
                for fig in child.result.figures.values():
                    plt.close(fig)
