import mlflow
from pydantic import validate_call

from dlkit.settings import MLflowClientSettings

TIMEOUT = 5  # seconds


def get_or_create_experiment(experiment_name: str) -> str:
    """Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

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
    """Initialize the MLflow client with the specified configuration.

    Ensures that the MLflow tracking URI points to a valid directory, and
    creates or retrieves the default experiment.

    Args:
        settings: The MLflow configuration.

    Returns:
        The experiment ID of the default experiment.
    """
    mlflow.set_tracking_uri(str(settings.tracking_uri))
    experiment_id = get_or_create_experiment(settings.experiment_name)
    return experiment_id
