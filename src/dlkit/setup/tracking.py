from time import sleep
import requests
import mlflow
from loguru import logger
from pydantic import validate_call
from dlkit.scripts.mlflow_server import start_server
from dlkit.settings import MLflowSettings


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


def try_start_server(settings: MLflowSettings) -> None:
    counter: int = 0
    while (
        not is_server_running(settings.client.tracking_uri)
        and counter < settings.client.max_trials
    ):
        counter += 1
        logger.error(f"MLflow server is not running at {settings.client.tracking_uri}")
        logger.info(f"Trying to start the MLflow server trial {counter}")
        process = start_server(settings.server)
        sleep(1)  # Wait for the server to start


@validate_call
def initialize_mlflow_client(settings: MLflowSettings) -> str:
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

    try_start_server(settings)

    if not is_server_running(settings.client.tracking_uri):
        logger.error(
            f"Failed to start the MLflow server after {settings.client.max_trials} trials."
        )
        raise ConnectionError("Failed to start the MLflow server.")

    logger.info("MLflow server is up.")
    mlflow.set_tracking_uri(settings.client.tracking_uri)
    experiment_id = get_or_create_experiment(settings.client.experiment_name)
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
        return False
