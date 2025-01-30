import numpy as np
from lightning.pytorch import seed_everything
import mlflow
import mlflow.pytorch

from dlkit.io.readers import load_config, parse_config_decorator
from dlkit.io.logging import get_logger
from dlkit.setup.tracking import initialize_mlflow_client
from dlkit.setup.tracking import MLFlowConfig
from dlkit.setup.datamodule import initialize_datamodule
from dlkit.setup.trainer import initialize_trainer
from dlkit.setup.model import initialize_model
from dlkit.scripts.start_mlflow_server import start_mlflow_server
import torch

logger = get_logger(__name__)
torch.set_float32_matmul_precision("medium")
seed_everything(1)


@parse_config_decorator
def main(config: dict):
    mlflow_config = MLFlowConfig(**config["mlflow"])

    server_process = start_mlflow_server(mlflow_config.server)
    experiment_id = initialize_mlflow_client(mlflow_config)

    data_module = initialize_datamodule(config)
    dataset = data_module.dataset
    trainer = initialize_trainer(config)
    model = initialize_model(config, data_module.dataset.shapes)
    dataset_source = mlflow.data.dataset_source.DatasetSource.from_dict(
        {
            "features": dataset.features_path,
            "targets": dataset.targets_path,
        }
    )

    # Function to handle termination signals
    # def terminate_server(signum, frame):
    #     logger.info("Received termination signal. Stopping MLflow server...")
    #     server_process.terminate()
    #     server_process.wait()
    #     exit(0)

    # Register the signal handler
    # signal.signal(signal.SIGINT, terminate_server)
    # signal.signal(signal.SIGTERM, terminate_server)

    # Start MLFlow run
    try:
        with mlflow.start_run(run_name=mlflow_config.run_name) as run:
            run_id = run.info.run_id
            mlflow.pytorch.autolog(log_models=True)
            mlflow.enable_system_metrics_logging()

            trainer.fit(model, datamodule=data_module)
            trainer.test(model, datamodule=data_module)
            predictions = trainer.predict(model, datamodule=data_module)

            # Convert predictions (list of Tensors) to a single NumPy array if possible
            # Assuming predictions is a list of tensors or arrays
            if isinstance(predictions, list) and len(predictions) > 0:
                predictions_np = torch.stack(predictions).numpy()
                np.save(config["paths"]["predictions"], predictions_np)
                mlflow.log_artifact(config["paths"]["predictions"])

            # Log the model
            mlflow_dataset = mlflow.data.from_numpy(
                dataset.features,
                targets=dataset.targets,
                source=dataset_source,
            )
            signature = mlflow.models.infer_signature(dataset.features, dataset.targets)
            mlflow.pytorch.log_model(model, "model", signature=signature)
            if mlflow_config.register_model:
                mlflow.register_model(
                    model_uri=f"runs:/{run_id}/model", name=config["model"]["name"]
                )
            mlflow.log_input(mlflow_dataset, "dataset")

            # Log hyperparameters
            mlflow.log_params(model.hparams)
            mlflow.log_dict(config["paths"], "paths.yml")
            mlflow.log_dict(dataset.indices, "splits.json")
            (
                mlflow.log_text(dataset.indices_path, "indices.json")
                if dataset.indices_path
                else None
            )

        logger.info(f"Training completed. Run ID: {run_id}")
    except Exception as e:
        raise e
    finally:
        # Ensure the server process is terminated
        if server_process:
            logger.info("Terminating MLflow server...")
            server_process.terminate()


if __name__ == "__main__":
    main()
