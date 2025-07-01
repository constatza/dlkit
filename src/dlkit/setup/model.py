from lightning.pytorch import LightningModule
from pytorch_forecasting import TimeSeriesDataSet

from dlkit.nn.primitives.base import PipelineNetwork
from dlkit.settings import ModelSettings
from dlkit.utils.loading import load_class
from dlkit.nn.primitives.graph import GraphNetwork
from dlkit.datasets import GraphDataset, BaseDataset


def build_model(
    *,
    settings: ModelSettings,
    dataset: BaseDataset,
) -> LightningModule:
    """Builds a LightningModule based on the provided settings and pipeline.

    Args:
        settings (ModelSettings): The settings object for the model.
        dataset (Dataset, optional): The dataset to use for the model. Defaults to None.

    Returns:
        LightningModule: The LightningModule for the model.
    """
    if isinstance(dataset, GraphDataset):
        return GraphNetwork(settings=settings.model_copy(update={"shape": dataset.shape}))

    if isinstance(dataset, TimeSeriesDataSet):
        class_name = load_class(settings.name, settings.module_path)
        applied_settings = settings.to_dict_compatible_with(class_name)
        return class_name(**applied_settings).from_dataset(
            dataset,
        )  # noqa: D100
    return PipelineNetwork(settings=settings.model_copy(update={"shape": dataset.shape}))
