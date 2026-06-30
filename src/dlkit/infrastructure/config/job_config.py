"""Job config — top-level discriminated union of training, inference, and search jobs."""

from __future__ import annotations

from pydantic import Field, model_validator

from dlkit.infrastructure.config.core.base_settings import BasicSettings
from dlkit.infrastructure.config.data_settings import DataSettings
from dlkit.infrastructure.config.experiment_settings import ExperimentSettings
from dlkit.infrastructure.config.model_components import ModelComponentSettings
from dlkit.infrastructure.config.run_settings import RunSettings
from dlkit.infrastructure.config.search_settings import SearchSettings
from dlkit.infrastructure.config.tracking_settings import TrackingSettings
from dlkit.infrastructure.config.training_settings import TrainingSettings


class JobConfig(BasicSettings):
    """Base job config. All workflow types share this structure.

    Args:
        run: Execution control (type, seed, precision, profile references).
        experiment: Experiment identity and registration metadata.
        model: Model class selector and hyperparameters.
        data: Dataset, dataloader, and DataModule configuration.
        training: Training pipeline configuration.
        search: HPO search configuration.
        tracking: Tracking backend connection.
    """

    run: RunSettings
    experiment: ExperimentSettings | None = None
    model: ModelComponentSettings | None = None
    data: DataSettings | None = None
    training: TrainingSettings | None = None
    search: SearchSettings | None = None
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)


class TrainingJobConfig(JobConfig):
    """Validated training job: model, data, and training sections are required.

    Args:
        model: Model class selector and hyperparameters (required).
        data: Dataset, dataloader, and DataModule configuration (required).
        training: Training pipeline configuration (required).
    """

    model: ModelComponentSettings
    data: DataSettings
    training: TrainingSettings


class InferenceJobConfig(JobConfig):
    """Validated inference job: only model is required (data optional for batch predict).

    Args:
        model: Model class selector and hyperparameters (required).
    """

    model: ModelComponentSettings

    @model_validator(mode="after")
    def _checkpoint_required(self) -> InferenceJobConfig:
        """Ensure a checkpoint path is provided for inference.

        Returns:
            Self, if validation passes.

        Raises:
            ValueError: If model.checkpoint is None.
        """
        if self.model.checkpoint is None:
            raise ValueError("InferenceJobConfig requires model.checkpoint to be set.")
        return self


class SearchJobConfig(JobConfig):
    """Validated HPO job: same as training plus a non-empty search section.

    Args:
        model: Model class selector and hyperparameters (required).
        data: Dataset, dataloader, and DataModule configuration (required).
        training: Training pipeline configuration (required).
        search: HPO search configuration (required, must have non-empty space).
    """

    model: ModelComponentSettings
    data: DataSettings
    training: TrainingSettings
    search: SearchSettings

    @model_validator(mode="after")
    def _space_required(self) -> SearchJobConfig:
        """Ensure the search space is non-empty.

        Returns:
            Self, if validation passes.

        Raises:
            ValueError: If search.space is empty.
        """
        if not self.search.space:
            raise ValueError("SearchJobConfig requires at least one entry in search.space.")
        return self
