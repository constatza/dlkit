# training.py
import inspect
import types

from lightning.pytorch import seed_everything
import mlflow
import mlflow.pytorch
import argparse

from dlkit.io.readers import load_config
from dlkit.io.logging import get_logger
from dlkit.setup.tracking import initialize_mlflow_client
from dlkit.setup.tracking import MLFlowConfig
from dlkit.setup.datamodule import initialize_datamodule
from dlkit.setup.trainer import initialize_trainer
from dlkit.setup.model import initialize_model
from dlkit.scripts.start_mlflow_server import start_mlflow_server

logger = get_logger(__name__)
seed_everything(1)


def main(config):
    mlflow_config = MLFlowConfig(**config["mlflow"])
    # Initialize data module and model
    data_module = initialize_datamodule(config)

    # Start MLflow server if not running
    server_process = start_mlflow_server(mlflow_config.server)
    # Initialize MLflow client
    experiment_id = initialize_mlflow_client(mlflow_config)
    trainer = initialize_trainer(config)
    model = initialize_model(config, data_module.shapes)

    mlflow.pytorch.autolog(log_models=True)
    # Start MLflow run
    with mlflow.start_run() as run:
        # Log hyperparameters
        # Enable MLflow autologging
        mlflow.log_params(model.hparams)

        # Train the model
        trainer.fit(model, datamodule=data_module)
        trainer.test(model, datamodule=data_module)
        mlflow.log_artifact(data_module.indices_path)
        run_id = run.info.run_id

    logger.info(f"Training completed. Run ID: {run_id}")
    server_process.terminate()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "config",
        type=str,
        help="Path to the training configuration file. Must NOT contain ranges for hyperparameters.",
    )
    args = argparser.parse_args()
    config = load_config(args.config)
    main(config)
