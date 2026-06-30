"""Regression tests: transform chains are fitted at build time, not via Lightning hooks.

Transform fitting used to be triggered by ``TransformFittingCallback.on_fit_start``,
a Lightning ``Callback``. Lightning's ``Tuner.lr_find()`` strips ``trainer.callbacks``
down to its own internal callback before running the LR-range-test scan loop, so the
callback never executed during tuning, leaving any feature/target entry with a
fittable transform (e.g. ``MinMaxScaler``) unfitted — causing
``TransformNotFittedError`` on the first scan batch and silently disabling LR tuning
every time.

Fitting now happens once, deterministically, in ``IBuildStrategy.build()`` — before
any ``Trainer``/``Tuner`` object exists at all. This removes the dependency on which
Lightning hooks survive ``Tuner``'s callback manipulation entirely: by the time
``BuildFactory().build_components(settings)`` returns, transforms are already fitted.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

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


def test_build_components_fits_transform_chain_before_any_trainer_exists(
    lr_tuning_settings_with_transform: TrainingJobConfig,
) -> None:
    """BuildFactory.build_components() fits transforms with no Trainer/Tuner involved.

    This is the root fix: fitting no longer depends on a Lightning Callback or
    LightningModule hook firing at the right time relative to Tuner.lr_find()'s
    internal callback stripping — it happens deterministically during the build
    phase, before any Trainer object is even constructed.
    """
    settings = lr_tuning_settings_with_transform
    components = BuildFactory().build_components(settings)
    model = components.model

    batch_transformer = cast(Any, model)._batch_transformer
    assert batch_transformer.is_fitted()


def test_build_components_is_a_noop_for_graph_strategy_with_no_transforms(
    graph_settings: TrainingJobConfig,
) -> None:
    """fit_transforms_if_needed() is a harmless no-op for GraphBuildStrategy.

    GraphLightningWrapper has no batch transformer at all (graph models work
    with raw PyG Data/Batch objects, not dlkit's TensorDict transform
    pipeline) — the build-phase fit hook in IBuildStrategy.build() must not
    raise when the model has no fittable batch transformer to find.
    """
    components = BuildFactory().build_components(graph_settings)

    assert not hasattr(components.model, "_batch_transformer")


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
