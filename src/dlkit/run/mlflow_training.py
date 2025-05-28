import mlflow
import pandas as pd
import polars as pl
from loguru import logger
from numpy import ndarray
from torch import Tensor
from torch.utils.data import Dataset
from lightning.pytorch import LightningDataModule

from dlkit.datamodules import InMemoryModule
from dlkit.datatypes.learning import TrainingState
from dlkit.run.vanilla_training import train_vanilla
from dlkit.settings import Settings
from dlkit.setup.mlflow_client import initialize_mlflow_client
from dlkit.setup.mlflow_server import ServerProcess

type Data = pl.DataFrame | pd.DataFrame | Tensor | ndarray

SETTINGS_FILENAME = "config.json"
IDX_SPLIT_FILENAME = "idx_split.json"


def train_mlflow(
    settings: Settings, datamodule: LightningDataModule | None = None
) -> TrainingState:
    """Trains, tests, and predicts using the provided configuration.
    This functions utilizes MLflow in order to monitor and track the
    training process, model performance, and other relevant metrics.

    Args:
        settings (Settings): The DLkit settings object.
        datamodule (LightningDataModule, optional): The datamodule to use for training.
            If not provided, it will be built from the settings.
    """
    with ServerProcess(settings.MLFLOW.server) as server:
        experiment_id = initialize_mlflow_client(settings.MLFLOW.client)

        # Start MLFlow run
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=settings.MLFLOW.client.run_name,
            nested=not server.is_active,  # Use nested runs if a server is already running
        ) as run:
            run_id = run.info.run_id
            mlflow.pytorch.autolog()
            mlflow.log_dict(settings.model_dump(), SETTINGS_FILENAME)

            training_state = train_vanilla(settings, datamodule=datamodule)
            model = training_state.model
            datamodule: InMemoryModule = training_state.datamodule
            mlflow.models.set_model(model)

            # Log the model
            features, targets = get_sample(datamodule.dataset.train)
            log_mlflow_dataset(features, settings.PATHS.features.as_uri())
            log_mlflow_dataset(targets, settings.PATHS.targets.as_uri())
            mlflow.log_dict(datamodule.idx_split.model_dump(), IDX_SPLIT_FILENAME)

            # signature = mlflow.models.infer_signature(
            #     datamodule.dataset.train[0][0].cpu().numpy(),
            #     datamodule.dataset.train[0][1].cpu().numpy(),
            # )
            # mlflow.pytorch.log_model(
            #     model,
            #     name=settings.MODEL.name,
            #     signature=signature,
            #     params=settings.MODEL.model_dump(exclude_none=True),
            # )
            if settings.MLFLOW.client.register_model:
                mlflow.register_model(
                    model_uri=f"runs:/{run_id}/model",
                    name=settings.MODEL.name,
                )
    return training_state


def get_sample(dataset: Dataset) -> Data:
    """Get a sample from the dataset.

    Args:
        dataset (Dataset): The dataset to sample from.

    Returns:
        Data: A sample from the dataset.
    """
    return dataset[0]


def log_mlflow_dataset(sample: Data, source: str) -> None:
    """ "Builds a MLflow dataset from a given dataset sample and logs it to MLflow.

    Args:
            sample (Data): The dataset sample to be logged.
        source (str): The source of the dataset.j

    Returns:
        mlflow.data.Dataset: The MLflow dataset.
    """
    dataset = classify_mlflow_dataset(sample, source=source)
    if dataset:
        mlflow.log_input(dataset)
        logger.info(f"Logged dataset of type {dataset.__class__.__name__} to MLflow.")
    else:
        logger.warning(f"Could not log dataset of type {sample.__class__.__name__} to MLFlow.")


def classify_mlflow_dataset(item: Data, **kwargs) -> mlflow.data.Dataset | None:
    """Classifies the dataset type and returns the corresponding MLflow dataset.

    Args:
        item (Data): The dataset to be classified.
    """
    if isinstance(item, Tensor):
        item = item.cpu().numpy()
    elif isinstance(item, pl.DataFrame):
        item = item.to_pandas()

    if isinstance(item, pd.DataFrame):
        return mlflow.data.from_pandas(item, **kwargs)
    elif isinstance(item, ndarray):
        return mlflow.data.from_numpy(item, **kwargs)
    else:
        return None
