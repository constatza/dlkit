"""Trial-level MLflow logging: readable param names and model-class tagging.

Covers three related fixes:
- Dotted hyperparameter keys (e.g. "model.num_layers") are logged to MLflow
  with only the top-level section prefix stripped, not the full path.
- Every trial run is tagged with the model class name at log-time, so a run
  can be identified even when its run name is uninformative.
- Trial metrics exclude train/val (already epoch-logged by MLflowEpochLogger)
  and pass test metrics through once, with no step, instead of duplicating
  them at step 0.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dlkit.engine.tracking.artifact_logger import TAG_MODEL_CLASS
from dlkit.engine.workflows.optimization.infrastructure.tracking import (
    log_trial_hyperparameters,
    log_trial_metrics,
    log_trial_settings,
)
from dlkit.engine.workflows.optimization.value_objects.models import Trial, TrialState
from dlkit.infrastructure.config.job_config import JobConfig
from dlkit.infrastructure.config.model_components import ModelComponentSettings
from dlkit.infrastructure.config.run_settings import RunSettings


@pytest.fixture
def trial() -> Trial:
    return Trial(
        trial_id="trial-0",
        trial_number=0,
        hyperparameters={},
        state=TrialState.RUNNING,
    )


@pytest.fixture
def run_context() -> MagicMock:
    return MagicMock()


def test_log_trial_hyperparameters_strips_only_top_level_section_prefix(
    trial: Trial, run_context: MagicMock
) -> None:
    hyperparameters = {
        "model.num_layers": 3,
        "training.optimizer.lr": 0.01,
    }

    log_trial_hyperparameters(hyperparameters, trial, run_context)

    run_context.log_params.assert_any_call(
        {
            "num_layers": 3,
            "optimizer.lr": 0.01,
        }
    )


@pytest.fixture
def settings_with_model() -> JobConfig:
    return JobConfig(
        run=RunSettings(type="train"),
        model=ModelComponentSettings(name="MyModel"),
    )


def test_log_trial_settings_tags_model_class(
    settings_with_model: JobConfig, run_context: MagicMock
) -> None:
    log_trial_settings(settings_with_model, run_context)

    run_context.set_tag.assert_any_call(TAG_MODEL_CLASS, "MyModel")


@pytest.fixture
def mixed_stage_trial_metrics() -> dict[str, float]:
    return {
        "train_loss": 0.9,
        "val/loss": 0.5,
        "test_loss": 0.3,
        "test/accuracy": 0.95,
    }


def test_log_trial_metrics_excludes_train_val_and_passes_test_through(
    mixed_stage_trial_metrics: dict[str, float], run_context: MagicMock
) -> None:
    log_trial_metrics(mixed_stage_trial_metrics, run_context)

    run_context.log_metrics.assert_called_once_with(
        {
            "test_loss": pytest.approx(0.3),
            "test/accuracy": pytest.approx(0.95),
        }
    )
