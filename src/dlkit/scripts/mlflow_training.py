import sys
import traceback

import numpy as np
from lightning.pytorch import seed_everything
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from dlkit.io.readers import load_config, parse_config
from dlkit.io.logging import get_logger
from dlkit.setup.tracking import initialize_mlflow_client
from dlkit.setup.tracking import MLFlowConfig
from dlkit.setup.datamodule import initialize_datamodule
from dlkit.setup.trainer import initialize_trainer
from dlkit.setup.model import initialize_model
import torch

logger = get_logger(__name__)
torch.set_float32_matmul_precision("medium")
seed_everything(1)
import mlflow
from typing import Optional


def train(config: dict):
    mlflow_config = MLFlowConfig(**config["mlflow"])

    # Initialize MLflow client and get experiment_id
    experiment_id = initialize_mlflow_client(mlflow_config)

    # Start MLFlow run
    with mlflow.start_run(run_name=mlflow_config.run_name) as run:
        logger.info("Training started.")

        datamodule = initialize_datamodule(config)
        trainer = initialize_trainer(config)
        dataset_source = mlflow.data.dataset_source.DatasetSource.from_dict(
            {
                "features": datamodule.features_path,
                "targets": datamodule.targets_path,
            }
        )

        run_id = run.info.run_id
        mlflow.pytorch.autolog(log_models=False)
        mlflow.log_dict(config, "config.yml")

        datamodule.setup(stage="fit")
        mlflow.log_dict(datamodule.idx_split_path, "splits.json")
        mlflow_dataset = mlflow.data.from_numpy(
            datamodule.features,
            targets=datamodule.targets,
            source=dataset_source,
        )
        signature = mlflow.models.infer_signature(
            datamodule.features, datamodule.targets
        )
        mlflow.log_input(mlflow_dataset, "dataset")

        model = initialize_model(config, datamodule.shapes)
        mlflow.log_params(model.hparams)
        trainer.fit(model, datamodule=datamodule, ckpt_path=mlflow_config.ckpt_path)
        trainer.test(model, datamodule=datamodule)
        predictions = trainer.predict(model, datamodule=datamodule)

        # Convert predictions (list of Tensors) to a single NumPy array if possible
        # Assuming predictions is a list of tensors or arrays
        if isinstance(predictions, list) and len(predictions) > 0:
            predictions_np = torch.cat(predictions, dim=0).numpy()
            np.save(config["paths"]["predictions"], predictions_np)
            mlflow.log_artifact(config["paths"]["predictions"])

        # Log the model
        mlflow.pytorch.log_model(model, "model", signature=signature)
        if mlflow_config.register_model:
            mlflow.register_model(
                model_uri=f"runs:/{run_id}/model", name=config["model"]["name"]
            )

    logger.info(f"Training completed. Run ID: {run_id}")


def main():
    try:
        config = parse_config(description="Training script.")
        train(config)
    except Exception as e:

        logger.error(e)
        logger.error(traceback.format_exc())
    finally:
        sys.exit(0)


if __name__ == "__main__":
    main()
