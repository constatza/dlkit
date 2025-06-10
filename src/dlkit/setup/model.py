from lightning.pytorch import LightningModule
from loguru import logger
from pydantic import FilePath

from dlkit.datasets import ForecastingDataset
from dlkit.nn.primitives.base import PipelineNetwork
from dlkit.settings import ModelSettings
from dlkit.utils.loading import load_class


def build_model(
    *,
    settings: ModelSettings,
    checkpoint: FilePath | None = None,
    dataset: ForecastingDataset | None = None,
) -> LightningModule:
    """Builds a LightningModule based on the provided settings and pipeline.

    Args:
        settings (ModelSettings): The settings object for the model.
        checkpoint (FilePath, optional): The path to a checkpoint file. Defaults to None.
        dataset (Dataset, optional): The dataset to use for the model. Defaults to None.

    Returns:
        LightningModule: The LightningModule for the model.
    """
    model = None
    if settings.module_path.startswith("dlkit.nn"):
        model = PipelineNetwork(settings=settings.model_copy(update={"shape": dataset.shape}))

    elif settings.module_path == "pytorch_forecasting":
        class_name = load_class(settings.name, settings.module_path)
        applied_settings = settings.to_dict_compatible_with(class_name)
        model = class_name(**applied_settings).from_dataset(
            dataset.timeseries,
        )  # noqa: D100

    if checkpoint and model:
        logger.info(f"Loading model from checkpoint: {checkpoint}")
        return model.__class__.load_from_checkpoint(checkpoint)

    raise ValueError(f"Unknown module path: {settings.module_path}")
