"""Canonical allowlist of DLKit classes safe to unpickle from trusted checkpoints.

PyTorch 2.6+ defaults ``torch.load()`` to ``weights_only=True``, which blocks
unpickling of custom classes unless explicitly registered via
``torch.serialization.add_safe_globals``. DLKit checkpoints embed Pydantic
settings and data-entry objects (e.g. inside Lightning hyperparameters), so
every class that may appear there must be registered once, here — not
re-enumerated per call site, which drifts.
"""

from typing import Any


def dlkit_safe_globals() -> list[type[Any]]:
    """Return every DLKit class that may be pickled inside a checkpoint.

    Returns:
        list[type[Any]]: Classes to pass to ``torch.serialization.add_safe_globals``.
    """
    from dlkit.infrastructure.config.core.base_settings import (
        BasicSettings,
        ComponentSettings,
        HyperParameterSettings,
    )
    from dlkit.infrastructure.config.data_entries import (
        AutoencoderTarget,
        Latent,
        Prediction,
    )
    from dlkit.infrastructure.config.data_settings import DataModuleSelector, DataSettings
    from dlkit.infrastructure.config.dataloader_settings import DataloaderSettings
    from dlkit.infrastructure.config.job_config import (
        InferenceJobConfig,
        JobConfig,
        SearchJobConfig,
        TrainingJobConfig,
    )
    from dlkit.infrastructure.config.lr_tuner_settings import LRTunerSettings
    from dlkit.infrastructure.config.model_components import (
        LossComponentSettings,
        MetricComponentSettings,
        ModelComponentSettings,
        WrapperComponentSettings,
    )
    from dlkit.infrastructure.config.optimizer_settings import (
        OptimizerSettings,
        SchedulerSettings,
    )
    from dlkit.infrastructure.config.paths_settings import PathsSettings
    from dlkit.infrastructure.config.run_settings import RunSettings
    from dlkit.infrastructure.config.split_settings import IndexSplitSettings
    from dlkit.infrastructure.config.tracking_settings import TrackingSettings
    from dlkit.infrastructure.config.training_settings import StoppingSettings, TrainingSettings

    return [
        # Base settings
        BasicSettings,
        ComponentSettings,
        HyperParameterSettings,
        TrainingSettings,
        StoppingSettings,
        PathsSettings,
        RunSettings,
        # JobConfig classes
        JobConfig,
        TrainingJobConfig,
        InferenceJobConfig,
        SearchJobConfig,
        # Model settings
        ModelComponentSettings,
        WrapperComponentSettings,
        MetricComponentSettings,
        LossComponentSettings,
        # Training settings
        OptimizerSettings,
        SchedulerSettings,
        # Data settings
        DataSettings,
        DataModuleSelector,
        IndexSplitSettings,
        DataloaderSettings,
        # Tracking
        TrackingSettings,
        # Other settings
        LRTunerSettings,
        # Data entry classes
        Latent,
        AutoencoderTarget,
        Prediction,
    ]
