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
from dlkit.setup.datamodule import initialize_datamodule
from dlkit.setup.trainer import initialize_trainer
from dlkit.setup.model import initialize_model
import torch

torch.set_float32_matmul_precision("medium")
seed_everything(1)


@click.command(
    "MLFlow Training",
    help="Trains, tests, and predicts using the provided configuration.",
)
@click.argument("config-path")
@validate_call
def train(config_path: FilePath) -> None:

    settings = load_validated_settings(config_path)

    datamodule = initialize_datamodule(settings.DATAMODULE, settings.PATHS)
    trainer = initialize_trainer(settings.TRAINER)
    # Initialize MLflow client and get experiment_id
    experiment_id = initialize_mlflow_client(settings.MLFLOW.client)
    # Start MLFlow run
    with mlflow.start_run(
        experiment_id=experiment_id, run_name=settings.MLFLOW.client.run_name
    ) as run:
        logger.info("Training started.")

        dataset_source = mlflow.data.dataset_source.DatasetSource.from_dict(
            {
                "features": settings.PATHS.features,
                "targets": settings.PATHS.targets,
            }
        )

        run_id = run.info.run_id
        mlflow.pytorch.autolog(log_models=False)
        mlflow.log_dict(settings.model_dump(), "config.toml")

        datamodule.setup(stage="fit")
        mlflow.log_dict(datamodule.idx_split, "splits.json")
        mlflow_dataset = mlflow.data.from_numpy(
            datamodule.features,
            targets=datamodule.targets,
            source=dataset_source,
        )
        signature = mlflow.models.infer_signature(
            datamodule.features, datamodule.targets
        )
        mlflow.log_input(mlflow_dataset, "dataset")

        model = initialize_model(settings.MODEL, datamodule.shape)
        mlflow.log_params(model.hparams)
        trainer.fit(model, datamodule=datamodule, ckpt_path=settings.PATHS.ckpt_path)
        trainer.test(model, datamodule=datamodule)
        predictions = trainer.predict(model, datamodule=datamodule)

        # Log the model
        mlflow.pytorch.log_model(model, "model", signature=signature)
        if settings.MLFLOW.client.register_model:
            mlflow.register_model(
                model_uri=f"runs:/{run_id}/model", name=settings.MODEL.name
            )

    logger.info(f"Training completed. Run ID: {run_id}")


def main():
    try:
        train()
    except Exception as e:
        logger.error(traceback.format_exc())
    finally:
        sys.exit(0)


if __name__ == "__main__":
    main()
