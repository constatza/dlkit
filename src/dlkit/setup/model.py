from lightning.pytorch import LightningModule
from pytorch_forecasting import TemporalFusionTransformer

from dlkit.datasets import ForecastingDataset
from dlkit.networks.blocks.base import PipelineNetwork
from dlkit.settings import ModelSettings
from dlkit.utils.loading import load_class


def build_model(
    settings: ModelSettings,
    settings_path: str | None = None,
    dataset: ForecastingDataset | None = None,
) -> LightningModule:
    """Builds a LightningModule based on the provided settings and pipeline.

    Args:
        settings (ModelSettings): The settings object for the model.
        settings_path (str, optional): The path to the settings file. Defaults to None.
        dataset (Dataset, optional): The dataset to use for the model. Defaults to None.

    Returns:
        LightningModule: The LightningModule for the model.
    """
    if settings.module_path != "pytorch_forecasting":
        return PipelineNetwork(settings=settings.model_copy(update={"shape": dataset.shape}))

    if settings.module_path == "pytorch_forecasting":
        class_name: type(TemporalFusionTransformer) = load_class(
            settings.name, settings.module_path, settings_path
        )  # noqa: D100
        applied_settings = settings.to_dict_compatible_with(class_name)
        return class_name(**applied_settings).from_dataset(
            dataset.timeseries,
        )  # noqa: D100

    raise ValueError(f"Unknown module path: {settings.module_path}")
