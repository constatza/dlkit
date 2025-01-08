import mlflow
from pydantic import BaseModel, Field, validate_call
from dlkit.io.logging import get_logger
from dlkit.utils.system_utils import ensure_local_directory

logger = get_logger(__name__)


class MLFlowServerConfig(BaseModel):
    """
    Pydantic model for the 'mlflow.server' configuration section.
    """

    host: str = Field(..., description="MLflow server host address.")
    port: int = Field(..., description="MLflow server port number.")
    backend_store_uri: str = Field(..., description="Backend store URI for MLflow.")
    default_artifact_root: str = Field(
        ..., description="Default artifact root directory or URI."
    )
    tracking_uri: str = Field(..., description="Tracking URI template for MLflow.")
    terminate_apps_on_port: bool = Field(
        False,
        description="Whether to kill any applications running on the specified port.",
    )


class MLFlowConfig(BaseModel):
    """Configuration model for MLflow experiment settings."""

    experiment_name: str = Field(..., description="Name of the MLflow experiment.")
    run_name: str = Field(..., description="Name of the MLflow run.")
    log_models: bool = Field(False, description="Whether to log models.")
    server: MLFlowServerConfig = Field(
        ..., description="MLflow server configuration block."
    )
    register_model: bool = Field(default=False, description="Whether to register the model.")


@validate_call
def get_or_create_experiment(
    experiment_name: str, artifact_root_uri: str = None
) -> str:
    """
    Retrieves or creates an MLflow experiment by name.

    Args:
        experiment_name (str): The name of the MLflow experiment.
        artifact_root_uri (str): The artifact root URI for the experiment.

    Returns:
        str: The experiment ID.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment:
        experiment_id = experiment.experiment_id
        logger.info(
            f"Using existing experiment '{experiment_name}' with ID {experiment_id}"
        )
        if experiment.artifact_location != artifact_root_uri:
            logger.warning(
                "The existing experiment's artifact location (%s) does not match the specified artifact root (%s).",
                experiment.artifact_location,
                artifact_root_uri,
            )
    else:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
        )
        logger.info(
            f"Created new experiment '{experiment_name}' with ID {experiment_id}"
        )

    # Set current run's experiment context
    mlflow.set_experiment(experiment_name)
    return experiment_id


@validate_call
def initialize_mlflow_client(config: MLFlowConfig) -> str:
    """
    Initializes the MLflow tracking server environment by setting up directories and creating experiments.

    Args:
        config (MLFlowServerConfig): Configuration for the MLflow server.

    Returns:
        str: The experiment ID of the configured experiment.
    """
    # Ensure directories exist if local
    tracking_uri = config.server.tracking_uri
    ensure_local_directory(tracking_uri)

    # Set the MLflow tracking URI for the server
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"Set MLflow tracking URI: {tracking_uri}")

    # Create or retrieve the default experiment
    experiment_id = get_or_create_experiment(config.experiment_name)
    return experiment_id
