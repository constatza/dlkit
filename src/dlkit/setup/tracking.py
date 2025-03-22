import sys
import requests
import mlflow
from loguru import logger
from pydantic import BaseModel, Field, validate_call
from dlkit.utils.system_utils import ensure_local_directory
from dlkit.settings import MLflowClientSettings


def get_or_create_experiment(experiment_name):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters:
    - experiment_name (str): Name of the MLflow experiment.

    Returns:
    - str: ID of the existing or newly created MLflow experiment.
    """

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(experiment_name)


@validate_call
def initialize_mlflow_client(settings: MLflowClientSettings) -> str:
    # Ensure directories exist if local
    """
    Initialize the MLflow client with the specified configuration.

    Ensures that the MLflow tracking URI points to a valid directory, and
    creates or retrieves the default experiment.

    Args:
        settings: The MLflow configuration.

    Returns:
        The experiment ID of the default experiment.
    """

    # ensure_local_directory(settings.tracking_uri)

    if is_server_running(settings.tracking_uri):
        logger.info("MLflow server is already running.")
    else:
        sys.exit(1)

    # Set the MLflow tracking URI for the server
    mlflow.set_tracking_uri(settings.tracking_uri)

    # Create or retrieve the default experiment
    experiment_id = get_or_create_experiment(settings.experiment_name)
    return experiment_id


def is_server_running(tracking_uri: str) -> bool:
    """
    Check if the server is running by querying its health endpoint.

    Args:
        tracking_uri (str): The tracking URI of the server.
    Returns:
        bool: True if the server responds with status 200, False otherwise.
    """
    url = f"{tracking_uri}/health"
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except requests.ConnectionError as e:
        logger.error(f"MLflow Connection Error: {e}")
        return False
