import argparse
import os
import json
import numpy as np
import mlflow
import mlflow.pytorch
from lightning.pytorch import seed_everything
from typing import Any, Dict

from dlkit.io.readers import parse_config_decorator
from dlkit.io.logging import get_logger
from dlkit.setup.tracking import initialize_mlflow_client
from dlkit.setup.tracking import MLFlowConfig
from dlkit.setup.datamodule import initialize_datamodule
from dlkit.setup.trainer import initialize_trainer
from dlkit.setup.model import initialize_model  # Used for shapes, if needed

logger = get_logger(__name__)
seed_everything(1)


def load_model_from_mlflow(run_id: str, model_artifact_path: str = "model") -> Any:
    """
    Load a PyTorch model from MLflow run artifacts.

    Args:
        run_id (str):
            The MLflow run ID from which to load the model.
        model_artifact_path (str):
            The relative artifact path where the model is stored. Defaults to "model".

    Returns:
        Any: The loaded PyTorch model.
    """
    # The model URI format for MLflow is 'runs:/{run_id}/{artifact_path}'
    model_uri = f"runs:/{run_id}/{model_artifact_path}"
    model = mlflow.pytorch.load_model(model_uri)
    return model


def predict_and_log(
        trainer: Any,
        model: Any,
        data_module: Any,
        run_id: str,
        artifact_subdir: str = "prediction_results",
) -> None:
    """
    Run predictions using the trainer and model, then log results as MLflow artifacts.

    Args:
        trainer (Any):
            The Lightning Trainer instance used for prediction.
        model (Any):
            The loaded model to predict with.
        data_module (Any):
            The datamodule providing the prediction dataset.
        run_id (str):
            The MLflow run ID under which artifacts are logged.
        artifact_subdir (str):
            Subdirectory for storing prediction artifacts in MLflow.
    """
    # Ensure we are in an MLflow run context. If not started, start a new one.
    with mlflow.start_run(run_id=run_id):
        # Perform predictions
        predictions = trainer.predict(model, datamodule=data_module)

        # Convert predictions (list of Tensors) to a single NumPy array if possible
        # Assuming predictions is a list of tensors or arrays
        predictions_np = np.concatenate(
            [p.detach().cpu().numpy() for p in predictions], axis=0
        )

        # Save predictions locally as .npy
        temp_dir = "temp_prediction_artifacts"
        os.makedirs(temp_dir, exist_ok=True)
        pred_file_path = os.path.join(temp_dir, "predictions.npy")
        np.save(pred_file_path, predictions_np)

        # Optionally, log some metadata about predictions
        # Log artifacts
        mlflow.log_artifacts(temp_dir, artifact_subdir)

        # Clean up local files if desired
        # import shutil
        # shutil.rmtree(temp_dir, ignore_errors=True)


@parse_config_decorator
def main(config: Dict[str, Any]) -> None:
    """
    Main function for loading a model from MLflow artifacts and performing predictions.

    Args:
        config (Dict[str, Any]):
            Configuration dictionary loaded from the specified config file.
    """
    mlflow_config = MLFlowConfig(**config["mlflow"])

    # Initialize MLflow client and get experiment_id as in training script
    experiment_id = initialize_mlflow_client(mlflow_config)

    # Run ID must be provided or known from the configuration file.
    # For example, assume that the config["mlflow"]["run_id"] contains the run ID
    # of the previously completed training run.
    run_id = mlflow_config.run_id
    if run_id is None:
        raise ValueError("No run_id specified in MLflow config. Cannot load model.")

    # Initialize data module (for prediction dataset)
    data_module = initialize_datamodule(config)
    data_module.setup(stage="predict")  # Ensure predict dataset is prepared if needed

    # Load the trained model from MLflow artifacts
    model = load_model_from_mlflow(run_id=run_id, model_artifact_path="model")

    # Initialize trainer for prediction
    trainer = initialize_trainer(config)

    # Perform prediction and log results
    predict_and_log(
        trainer, model, data_module, run_id=run_id, artifact_subdir="predictions"
    )


if __name__ == "__main__":
    main()
