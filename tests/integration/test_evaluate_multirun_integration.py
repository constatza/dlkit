"""End-to-end test: train a real multirun sweep, then batch-evaluate its children.

Verifies ``evaluate_multirun()`` fans a single ``evaluate()`` call out over
every child run of a real ``MultiRunOrchestrator`` sweep, using genuine
checkpoints/MLflow runs (not mocked), plus a focused unit-level test that
``evaluate_multirun()`` fails loudly when the parent run has no children.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from dlkit.common.errors import WorkflowError
from dlkit.common.results import MultiRunResult
from dlkit.engine.tracking.mlflow_tracker import MLFLOW_DEFAULT_EXPERIMENT, MLflowTracker
from dlkit.engine.tracking.run_queries import find_child_run_ids
from dlkit.engine.training.vanilla_executor import VanillaExecutor
from dlkit.engine.workflows.factories.build_factory import BuildFactory
from dlkit.engine.workflows.multi_run import MultiRunOrchestrator, RunVariant
from dlkit.infrastructure.config.job_config import InferenceJobConfig, TrainingJobConfig
from dlkit.infrastructure.config.tracking_settings import TrackingSettings
from dlkit.interfaces.inference.evaluate_multirun import ChildEvaluation, evaluate_multirun

# Same model/data shape as tests/integration/conftest.py's _make_training_job_config,
# since InferenceJobConfig must describe the identical model + data the checkpoints
# were trained with (component "name"/"class" is write-only in the settings models, so
# it can't be recovered by dumping a validated TrainingJobConfig back out).
FEATURE_SIZE = 4
NUM_VARIANTS = 3


def _build_inference_settings(minimal_dataset: dict[str, Path]) -> InferenceJobConfig:
    """Build an InferenceJobConfig matching the FFNN trained by the sweep variants.

    ``model.checkpoint`` is required by ``InferenceJobConfig`` but unused here: the
    per-child checkpoint resolved via ``run_checkpoint`` always takes precedence over
    ``settings.model.checkpoint`` inside ``evaluate()``.
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

    A `run_checkpoint`-resolved checkpoint is downloaded to an arbitrary temp
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
    orchestrator = MultiRunOrchestrator(BuildFactory(), VanillaExecutor(), tracker)

    variants = [
        RunVariant(settings=settings, run_name=f"variant-{i}")
        for i, settings in enumerate(sweep_variant_settings)
    ]

    parent_run_ids: list[str] = []
    orchestrator.run_sweep(
        variants=variants,
        # `MultiRunOrchestrator._run_one()` opens each child run without forwarding
        # `experiment_name`, so `MLflowResourceManager.create_run()` places every
        # child under its own default (`MLFLOW_DEFAULT_EXPERIMENT`) rather than the
        # parent's experiment — even though the `mlflow.parentRunId` tag is still set
        # correctly. `find_child_run_ids()` only searches within the parent run's own
        # experiment, so parent and children must share one here for discovery to
        # work; naming the parent's experiment `MLFLOW_DEFAULT_EXPERIMENT` makes the
        # child's implicit default land in the same place. This is a real
        # interoperability gap between `MultiRunOrchestrator` and `find_child_run_ids`,
        # not something this test should paper over silently — see the task report.
        experiment_name=MLFLOW_DEFAULT_EXPERIMENT,
        parent_run_name="sweep_parent",
        on_sweep_complete=lambda parent_run, _results: parent_run_ids.append(parent_run.run_id),
    )

    return parent_run_ids[0], sweep_variant_settings


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
        assert result.parent_run_id == parent_run_id
        assert len(result.children) == NUM_VARIANTS
        assert all(isinstance(child, ChildEvaluation) for child in result.children)
        assert {child.run_id for child in result.children} == expected_run_ids
        for child in result.children:
            assert child.result.metrics.keys() == {"mae", "rmse", "r2"}
            assert all(isinstance(v, float) for v in child.result.metrics.values())
            assert child.result.predictions.shape[0] == child.result.targets.shape[0] > 0
            # Each ChildEvaluation wasn't itself logged to MLflow (log_to_mlflow
            # defaults to False), so its own mlflow_run_id must stay unset —
            # distinct from `child.run_id`, the source checkpoint's run.
            assert child.result.mlflow_run_id is None
    finally:
        for child in result.children:
            for fig in child.result.figures.values():
                plt.close(fig)


def test_evaluate_multirun_with_log_to_mlflow_distinguishes_source_and_logging_runs(
    trained_sweep: tuple[str, tuple[TrainingJobConfig, ...]],
    multirun_inference_settings: InferenceJobConfig,
) -> None:
    """`ChildEvaluation.run_id` (source checkpoint's run) and its own
    `EvaluationResult.mlflow_run_id` (the run opened to *log* that evaluation, only
    present when `log_to_mlflow=True`) must be two genuinely distinct, non-empty run
    ids — not merely two fields that happen to both be unset.
    """
    parent_run_id, _variant_settings = trained_sweep

    result = evaluate_multirun(
        multirun_inference_settings, parent_run_id=parent_run_id, log_to_mlflow=True
    )

    try:
        assert len(result.children) == NUM_VARIANTS
        for child in result.children:
            assert child.run_id
            assert child.result.mlflow_run_id
            assert child.run_id != child.result.mlflow_run_id
    finally:
        for child in result.children:
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
        "dlkit.interfaces.inference.evaluate_multirun.find_child_run_ids",
        _raise_no_children,
    )

    settings = _build_inference_settings(minimal_dataset)

    with pytest.raises(WorkflowError, match="no active child runs"):
        evaluate_multirun(settings, parent_run_id="lonely-parent")
