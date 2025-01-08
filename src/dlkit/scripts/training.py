from lightning.pytorch import seed_everything
import mlflow
import mlflow.pytorch
import argparse
import signal

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

def train(config):
    mlflow_config = MLFlowConfig(**config["mlflow"])

    server_process = start_mlflow_server(mlflow_config.server)
    experiment_id = initialize_mlflow_client(mlflow_config)

    data_module = initialize_datamodule(config)
    trainer = initialize_trainer(config)
    model = initialize_model(config, data_module.shapes)
    dataset_source = mlflow.data.dataset_source.DatasetSource.from_dict(
        {"features": data_module.features_path, "targets": data_module.targets_path}
    )

    # Function to handle termination signals
    def terminate_server(signum, frame):
        logger.info("Received termination signal. Stopping MLflow server...")
        server_process.terminate()
        server_process.wait()
        exit(0)

    # Register the signal handler
    signal.signal(signal.SIGINT, terminate_server)
    signal.signal(signal.SIGTERM, terminate_server)

    # Start MLFlow run
    try:
        with mlflow.start_run(run_name=mlflow_config.run_name) as run:
            run_id = run.info.run_id
            mlflow.pytorch.autolog(log_models=True)
            mlflow.enable_system_metrics_logging()
            # Log hyperparameters
            # Enable MLflow autologging
            mlflow.log_params(model.hparams)
            mlflow.log_dict(config["paths"], "paths.yml")
            mlflow.log_dict(config["model"], "model_params.yml")

            # Train the model
            trainer.fit(model, datamodule=data_module)
            trainer.test(model, datamodule=data_module)
            dataset = mlflow.data.from_numpy(
                data_module.features, targets=data_module.targets, source=dataset_source
            )
            signature = mlflow.models.infer_signature(
                data_module.features, data_module.targets
            )
            mlflow.pytorch.log_model(model, "model", signature=signature)
            if mlflow_config.register_model:
                mlflow.register_model(
                    model_uri=f"runs:/{run_id}/model", name=config["model"]["name"]
                )
            mlflow.log_input(dataset, "dataset")
            mlflow.log_artifact(data_module.indices_path)

        logger.info(f"Training completed. Run ID: {run_id}")
    except Exception as e:
        raise e
    finally:
        # Ensure the server process is terminated
        if server_process:
            logger.info("Terminating MLflow server...")
            server_process.terminate()
            server_process.wait()


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "config",
        type=str,
        help="Path to the training configuration file. Must NOT contain ranges for hyperparameters.",
    )
    args = argparser.parse_args()
    config = load_config(args.config)
    train(config)

if __name__=="__main__":
    main()