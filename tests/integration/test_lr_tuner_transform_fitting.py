"""Regression tests: LR tuning must fit transform chains before Lightning's LR scan.

Lightning's ``Tuner.lr_find()`` strips ``trainer.callbacks`` down to its own
internal callback before running the LR-range-test scan loop, so dlkit's
``TransformFittingCallback.on_fit_start`` never executes during tuning. Any
feature/target entry with a fittable transform (e.g. ``MinMaxScaler``) was
therefore left unfitted, causing ``TransformNotFittedError`` on the first scan
batch and silently disabling LR tuning every time.

``num_training=2`` and ``max_lr=1e-3`` keep the scan minimal and fast: this
test only needs to prove the transform gets fitted before the scan touches it,
not that Lightning's own ``suggestion()`` heuristic (which needs more points
than that to compute a gradient) succeeds.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any, cast

import pytest

from dlkit.engine.training.tuning import LRTuner
from dlkit.engine.workflows.factories.build_factory import BuildFactory
from dlkit.infrastructure.config.job_config import TrainingJobConfig
from dlkit.infrastructure.config.lr_tuner_settings import LRTunerSettings
from dlkit.infrastructure.config.trainer_settings import TrainerSettings
from dlkit.infrastructure.config.transform_settings import TransformSettings


def _with_fittable_feature_transform(training_settings: TrainingJobConfig) -> TrainingJobConfig:
    """Attach a real, unfitted MinMaxScaler to the feature entry and enable LR tuning.

    ``fast_dev_run`` is disabled because Lightning's LR finder is a no-op under
    ``fast_dev_run`` (it skips itself entirely), which would mask this bug.

    Args:
        training_settings: Base integration training settings fixture.

    Returns:
        TrainingJobConfig with a fittable feature transform and lr_tuner enabled.
    """
    minmax = TransformSettings(
        name="MinMaxScaler", module_path="dlkit.domain.transforms.minmax", dim=0
    )
    data = training_settings.data
    feature = data.features[0].model_copy(update={"transforms": [minmax]})
    updated_data = data.model_copy(update={"features": (feature, *data.features[1:])})
    updated_trainer = TrainerSettings.model_validate(
        {
            "fast_dev_run": False,
            "enable_checkpointing": False,
            "accelerator": "cpu",
            "enable_progress_bar": False,
            "enable_model_summary": False,
            # Only bounds the real post-tuning training run (api_train calls
            # trainer.fit() again after LR tuning); the LR scan itself is
            # bounded separately by lr_tuner.num_training.
            "max_epochs": 1,
        }
    )
    updated_training = training_settings.training.model_copy(
        update={
            "lr_tuner": LRTunerSettings(max_lr=1e-3, num_training=2),
            "trainer": updated_trainer,
        }
    )
    return training_settings.model_copy(update={"data": updated_data, "training": updated_training})


@pytest.fixture
def lr_tuning_settings_with_transform(training_settings: TrainingJobConfig) -> TrainingJobConfig:
    """Training settings with a fittable feature transform and LR tuner enabled."""
    return _with_fittable_feature_transform(training_settings)


def test_lr_tuner_fits_transform_chain_before_scanning(
    lr_tuning_settings_with_transform: TrainingJobConfig,
) -> None:
    """LRTuner.tune() must fit unfitted transform chains before Lightning's LR scan.

    Calls LRTuner directly (bypassing VanillaExecutor._apply_lr_tuning's broad
    except-Exception swallow) so the regression surfaces as a real failure.
    A RuntimeError("...failed to suggest a learning rate...") is an accepted,
    unrelated outcome here: num_training=2 is too small for Lightning's own
    suggestion() heuristic (it discards the first 10 and last 1 points) to
    compute a gradient. Only TransformNotFittedError indicates this bug.
    """
    settings = lr_tuning_settings_with_transform
    components = BuildFactory().build_components(settings)
    model = components.model
    datamodule = components.datamodule
    lr_tuner_settings = settings.training.lr_tuner
    assert lr_tuner_settings is not None

    batch_transformer = cast(Any, model)._batch_transformer
    assert not batch_transformer.is_fitted()

    tuning_trainer = settings.training.trainer.build(session=None)
    with suppress(RuntimeError):
        LRTuner().tune(tuning_trainer, model, lr_tuner_settings, datamodule)

    assert batch_transformer.is_fitted()


def test_apply_lr_tuning_does_not_silently_swallow_transform_fitting_failure(
    lr_tuning_settings_with_transform: TrainingJobConfig,
) -> None:
    """End-to-end: api_train must never warn about an unfitted TransformChain.

    VanillaExecutor._apply_lr_tuning catches Exception broadly around the whole
    tuning call and logs a warning, so this is the only way the regression
    would otherwise be caught from the outside.
    """
    from loguru import logger

    from dlkit.interfaces.api import train as api_train

    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    try:
        api_train(lr_tuning_settings_with_transform)
    finally:
        logger.remove(sink_id)

    assert not any("TransformChain must be fitted" in m for m in messages)
