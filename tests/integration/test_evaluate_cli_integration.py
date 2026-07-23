"""End-to-end CLI tests: train real checkpoints/sweeps, evaluate through the CLI.

Distinct from `tests/integration/test_evaluate_integration.py` (which drives
`evaluate()` directly) and `tests/interfaces/cli/commands/test_evaluate_command_good_path.py`
(which mocks `evaluate_api` to isolate CLI flag translation) — this exercises
the real `dlkit evaluate --run-id`/`--latest-run` and `dlkit evaluate-multirun`
commands, through `CliRunner`, against genuine trained checkpoints and MLflow
runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from dlkit.engine.tracking.mlflow_tracker import MLflowTracker
from dlkit.engine.workflows.entrypoints import execute
from dlkit.engine.workflows.multi_run import MultiRunOrchestrator, RunSpec
from dlkit.infrastructure.config.job_config import TrainingJobConfig
from dlkit.infrastructure.config.tracking_settings import TrackingSettings
from dlkit.interfaces.api.functions import train as api_train
from dlkit.interfaces.cli.app import app as cli_app

# Same model/data shape as tests/integration/conftest.py's _make_training_job_config,
# since the CLI-side InferenceJobConfig TOML must describe the identical model + data
# the checkpoints were trained with (component "name" is write-only in the settings
# models, so it can't be recovered by dumping a validated TrainingJobConfig back out).
FEATURE_SIZE = 4
NUM_VARIANTS = 3


@pytest.fixture
def cli_runner() -> CliRunner:
    """Typer CLI test runner fixture with colors disabled.

    `tests/interfaces/cli/conftest.py` defines the same fixture, but its
    scope doesn't reach `tests/integration/`, so it's redefined locally here.
    """
    return CliRunner(
        env={
            "NO_COLOR": "1",
            "CLICOLOR": "0",
            "FORCE_COLOR": "0",
            "PY_COLORS": "0",
            "RICH_FORCE_TERMINAL": "0",
            "TERM": "dumb",
        }
    )


def _inference_config_toml(
    *,
    feature_path: Path,
    target_path: Path,
    tracking_uri: str,
    split_filepath: Path,
) -> str:
    """Build a TOML InferenceJobConfig matching the trained FFNN/FlexibleDataset shape.

    `model.checkpoint` is set to an unused placeholder: the CLI's `--run-id`/
    `--latest-run` resolved checkpoint always takes precedence over
    `settings.model.checkpoint` inside `evaluate()`.

    `data.splits.filepath` is set explicitly because a `--run-id`/`--latest-run`
    resolved checkpoint downloads to an arbitrary temp directory, not colocated
    next to a `splits/` directory the way a local training-output checkpoint is
    — so `evaluate()`'s default colocated-split auto-location can't apply
    (same rough edge `test_evaluate_integration.py` and
    `test_evaluate_multirun_integration.py` work around).
    """
    return f"""
[run]
type = "predict"

[experiment]
name = "integration_test"

[model]
name = "FFNN"
module_path = "dlkit.domain.nn"
hidden_size = {FEATURE_SIZE}
num_layers = 0
checkpoint = "unused-placeholder.ckpt"

[tracking]
backend = "mlflow"
uri = "{tracking_uri}"

[data]
name = "FlexibleDataset"
module_path = "dlkit.engine.data.datasets"
batch_size = 4
num_workers = 0
shuffle = false
pin_memory = false
persistent_workers = false

[[data.features]]
name = "x"
path = "{feature_path.as_posix()}"
format = "npy"

[[data.targets]]
name = "y"
path = "{target_path.as_posix()}"
format = "npy"

[data.splits]
filepath = "{split_filepath.as_posix()}"
"""


def _split_filepath(training_settings: TrainingJobConfig) -> Path:
    """The single `splits/*.json` file training persisted for this run."""
    training_cfg = training_settings.training
    assert training_cfg is not None
    trainer_cfg = training_cfg.trainer
    assert trainer_cfg is not None
    split_files = list(Path(trainer_cfg.default_root_dir).glob("splits/*.json"))
    assert len(split_files) == 1, f"expected exactly one split file, found {split_files}"
    return split_files[0]


@pytest.fixture
def training_settings_with_checkpoint_and_mlflow(
    training_settings: TrainingJobConfig, tmp_path: Path
) -> TrainingJobConfig:
    """Real checkpointing plus a real local (sqlite) MLflow tracking backend.

    Mirrors `test_evaluate_integration.py`'s fixture of the same name — run-based
    checkpoint selection needs an actual queryable MLflow run, not just a
    checkpoint file on local disk.
    """
    training_cfg = training_settings.training
    assert training_cfg is not None
    trainer_cfg = training_cfg.trainer
    assert trainer_cfg is not None

    mlruns_dir = tmp_path / "cli_run_checkpoint_mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow_uri = f"sqlite:///{(mlruns_dir / 'mlflow.db').as_posix()}"

    return training_settings.model_copy(
        update={
            "training": training_cfg.model_copy(
                update={
                    "trainer": trainer_cfg.model_copy(
                        update={
                            "enable_checkpointing": True,
                            "default_root_dir": tmp_path / "training_output",
                            "fast_dev_run": False,
                        }
                    )
                }
            ),
            "tracking": TrackingSettings(backend="mlflow", uri=mlflow_uri),
        }
    )


@pytest.fixture
def trained_cli_checkpoint(
    training_settings_with_checkpoint_and_mlflow: TrainingJobConfig,
) -> tuple[str, TrainingJobConfig]:
    """Train a real checkpoint/MLflow run, returning it alongside its settings.

    Training must complete (and persist its `splits/*.json` file) before
    `_split_filepath` can locate it, so this fixture — not the individual
    tests — owns calling `api_train`.
    """
    trained = api_train(training_settings_with_checkpoint_and_mlflow)
    assert trained.mlflow_run_id is not None
    return trained.mlflow_run_id, training_settings_with_checkpoint_and_mlflow


@pytest.fixture
def evaluate_config_path(
    minimal_dataset: dict[str, Path],
    trained_cli_checkpoint: tuple[str, TrainingJobConfig],
    tmp_path: Path,
) -> Path:
    """Write an InferenceJobConfig TOML file matching the trained checkpoint's shape."""
    _run_id, training_settings_with_checkpoint_and_mlflow = trained_cli_checkpoint
    config_content = _inference_config_toml(
        feature_path=minimal_dataset["features"],
        target_path=minimal_dataset["targets"],
        tracking_uri=training_settings_with_checkpoint_and_mlflow.tracking.uri,
        split_filepath=_split_filepath(training_settings_with_checkpoint_and_mlflow),
    )
    config_path = tmp_path / "evaluate.toml"
    config_path.write_text(config_content)
    return config_path


def test_evaluate_cli_with_run_id_evaluates_real_checkpoint(
    cli_runner: CliRunner,
    trained_cli_checkpoint: tuple[str, TrainingJobConfig],
    evaluate_config_path: Path,
) -> None:
    run_id, _training_settings = trained_cli_checkpoint

    result = cli_runner.invoke(
        cli_app,
        ["evaluate", "entry", str(evaluate_config_path), "--run-id", run_id],
    )

    assert result.exit_code == 0, result.output
    assert "Evaluation completed successfully" in result.output
    assert "mae" in result.output.lower()


def test_evaluate_cli_with_latest_run_evaluates_real_checkpoint(
    cli_runner: CliRunner,
    trained_cli_checkpoint: tuple[str, TrainingJobConfig],
    evaluate_config_path: Path,
) -> None:
    result = cli_runner.invoke(
        cli_app,
        ["evaluate", "entry", str(evaluate_config_path), "--latest-run"],
    )

    assert result.exit_code == 0, result.output
    assert "Evaluation completed successfully" in result.output
    assert "mae" in result.output.lower()


def test_evaluate_cli_rejects_run_id_and_latest_run_together(
    cli_runner: CliRunner,
    trained_cli_checkpoint: tuple[str, TrainingJobConfig],
    evaluate_config_path: Path,
) -> None:
    run_id, _training_settings = trained_cli_checkpoint

    result = cli_runner.invoke(
        cli_app,
        [
            "evaluate",
            "entry",
            str(evaluate_config_path),
            "--run-id",
            run_id,
            "--latest-run",
        ],
    )

    assert result.exit_code == 1
    assert "Pass at most one checkpoint-selection method" in result.output
    assert "Traceback" not in result.output


@pytest.fixture
def sweep_variant_settings(
    training_settings: TrainingJobConfig, tmp_path: Path
) -> tuple[TrainingJobConfig, ...]:
    """NUM_VARIANTS training configs sharing data/seed with a shared real MLflow backend.

    Mirrors `test_evaluate_multirun_integration.py`'s fixture of the same name.
    """
    training_cfg = training_settings.training
    assert training_cfg is not None
    trainer_cfg = training_cfg.trainer
    assert trainer_cfg is not None

    mlruns_dir = tmp_path / "cli_sweep_mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    tracking = TrackingSettings(
        backend="mlflow", uri=f"sqlite:///{(mlruns_dir / 'mlflow.db').as_posix()}"
    )

    def _variant(index: int) -> TrainingJobConfig:
        root = tmp_path / f"cli_sweep_output_{index}"
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
def trained_cli_sweep(
    sweep_variant_settings: tuple[TrainingJobConfig, ...],
) -> tuple[str, tuple[TrainingJobConfig, ...]]:
    """Train `sweep_variant_settings` via a real MultiRunOrchestrator.

    Returns:
        Tuple of (parent_run_id, sweep_variant_settings).
    """
    tracker = MLflowTracker()
    tracker.configure(sweep_variant_settings[0].tracking)
    orchestrator = MultiRunOrchestrator(tracker, execute)

    children = [
        RunSpec(
            id=f"cli-variant-{i}",
            label=f"cli-variant-{i}",
            settings=settings,
            run_name=f"cli-variant-{i}",
        )
        for i, settings in enumerate(sweep_variant_settings)
    ]

    result = orchestrator.run_sweep(
        children=children,
        experiment_name="cli_sweep_experiment",
        parent_run_name="cli_sweep_parent",
    )

    return result.parent_run_id, sweep_variant_settings


@pytest.fixture
def evaluate_multirun_config_path(
    minimal_dataset: dict[str, Path],
    trained_cli_sweep: tuple[str, tuple[TrainingJobConfig, ...]],
    tmp_path: Path,
) -> Path:
    """Write an InferenceJobConfig TOML file matching the sweep's model/data shape."""
    _parent_run_id, variant_settings = trained_cli_sweep
    config_content = _inference_config_toml(
        feature_path=minimal_dataset["features"],
        target_path=minimal_dataset["targets"],
        tracking_uri=variant_settings[0].tracking.uri,
        split_filepath=_split_filepath(variant_settings[0]),
    )
    config_path = tmp_path / "evaluate_multirun.toml"
    config_path.write_text(config_content)
    return config_path


def test_evaluate_multirun_cli_evaluates_every_child_run(
    cli_runner: CliRunner,
    trained_cli_sweep: tuple[str, tuple[TrainingJobConfig, ...]],
    evaluate_multirun_config_path: Path,
) -> None:
    parent_run_id, _variant_settings = trained_cli_sweep

    result = cli_runner.invoke(
        cli_app,
        [
            "evaluate-multirun",
            "entry",
            str(evaluate_multirun_config_path),
            "--parent-run-id",
            parent_run_id,
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Multirun evaluation completed successfully" in result.output
    assert "Parent run:" in result.output
    assert result.output.count("✅ success") == NUM_VARIANTS
