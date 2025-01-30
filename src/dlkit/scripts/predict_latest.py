import mlflow
import torch
from mlflow.tracking import MlflowClient

from dlkit.io.readers import load_config
from dlkit.setup.datamodule import initialize_datamodule
from dlkit.setup.trainer import initialize_trainer


# Fetch latest run automatically
def get_latest_run(experiment_name="Default"):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    latest_run = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )[0]
    return latest_run.info.run_id


# Fetch dataset and model
def load_artifacts_and_model(run_id, input_path: str, output_path: str = None):
    # Download artifacts
    input_path = mlflow.artifacts.download_artifacts(
        artifact_path=input_path, run_id=run_id
    )
    if output_path is not None:
        output_path = mlflow.artifacts.download_artifacts(
            artifact_path=output_path, run_id=run_id)

    datamodule = initialize_datamodule(input_path, output_path)
    trainer = initialize_trainer(datamodule)
    # Load datasets

    # Load model
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pytorch.load_model(model_uri)

    return train_data, train_labels, model


# Main workflow
if __name__ == "__main__":
    # Get the latest run ID
    config = load_config()
    latest_run_id = get_latest_run()

    # Load artifacts and model
    train_data, train_labels, model = load_artifacts_and_model(latest_run_id, config)

    # Convert data to PyTorch tensors
    train_data_tensor = torch.from_numpy(train_data).float()

    # Run predictions
    model.eval()
    with torch.no_grad():
        predictions = model(train_data_tensor)

    print("Predictions:")
    print(predictions)
