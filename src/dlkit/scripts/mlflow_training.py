import sys
import traceback

import numpy as np
import mlflow
from pydantic import validate_call, FilePath
from loguru import logger
from lightning.pytorch import seed_everything

import click
from dlkit.settings import Settings
from dlkit.io.settings import load_validated_settings
from dlkit.setup.tracking import initialize_mlflow_client
from dlkit.scripts.training import train
import torch

torch.set_float32_matmul_precision("medium")
seed_everything(1)


@validate_call
def train_mlflow(settings: Settings) -> None:
    """Trains, tests, and predicts using the provided configuration.
    This functions utilizes MLflow in order to monitor and track the
    training process, model performance, and other relevant metrics.

    Args:
        Settings: The DLkit settings object.
    """
    logger.info("Training started.")
    experiment_id = initialize_mlflow_client(settings.MLFLOW)
    # Start MLFlow run
    with mlflow.start_run(
        experiment_id=experiment_id, run_name=settings.MLFLOW.client.run_name
    ) as run:

        dataset_source = mlflow.data.dataset_source.DatasetSource.from_dict(
            {
                "features": settings.PATHS.features,
                "targets": settings.PATHS.targets,
            }
        )

        run_id = run.info.run_id
        mlflow.pytorch.autolog(log_models=False)
        mlflow.log_dict(settings.model_dump(), "config.toml")

        training_state = train(settings)
        model = training_state.model
        datamodule = training_state.datamodule

        mlflow.log_params(model.hparams)
        mlflow.log_dict(datamodule.idx_split, "idx_split.json")
        mlflow_dataset = mlflow.data.from_numpy(
            datamodule.features,
            targets=datamodule.targets,
            source=dataset_source,
        )
        signature = mlflow.models.infer_signature(
            datamodule.features, datamodule.targets
        )
        mlflow.log_input(mlflow_dataset, "dataset")
        mlflow.pytorch.log_model(model, "model", signature=signature)
        if settings.MLFLOW.client.register_model:
            mlflow.register_model(
                model_uri=f"runs:/{run_id}/model", name=settings.MODEL.name
            )

        logger.info(f"Training completed. Run ID: {run_id}")


@click.command(
    "MLFlow Training",
    help="Trains, tests, and predicts using the provided configuration.",
)
@click.argument("config-path")
def train_mlflow_cli(config_path: str):
    settings = load_validated_settings(config_path)
    train_mlflow(settings)


def main():
    try:
        train_mlflow_cli()
    except Exception as e:
        logger.error(traceback.format_exc())
    finally:
        sys.exit(0)


if __name__ == "__main__":
    main()
