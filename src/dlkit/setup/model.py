from lightning.pytorch import LightningModule

from dlkit.datasets import ForecastingDataset
from dlkit.nn.primitives.base import PipelineNetwork
from dlkit.settings import ModelSettings
from dlkit.utils.loading import load_class


def build_model(
    *,
    settings: ModelSettings,
    dataset: ForecastingDataset | None = None,
) -> LightningModule:
    """Builds a LightningModule based on the provided settings and pipeline.

    Args:
        settings (ModelSettings): The settings object for the model.
        dataset (Dataset, optional): The dataset to use for the model. Defaults to None.

    Returns:
        LightningModule: The LightningModule for the model.
    """
    if settings.module_path.startswith("dlkit.nn"):
        return PipelineNetwork(settings=settings.model_copy(update={"shape": dataset.shape}))
    else:
        class_name = load_class(settings.name, settings.module_path)
        applied_settings = settings.to_dict_compatible_with(class_name)
        return class_name(**applied_settings).from_dataset(
            dataset.timeseries,
        )  # noqa: D100
