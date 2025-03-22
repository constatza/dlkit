from pydantic import Field
from .base_settings import BaseSettings


class MLflowServerSettings(BaseSettings):
    scheme: str = Field(default="http", description="MLflow server scheme.")
    host: str = Field(default="127.0.0.1", description="MLflow server host address.")
    port: int = Field(default=5000, description="MLflow server port number.")
    backend_store_uri: str = Field(..., description="URI for the backend store.")
    default_artifact_root: str = Field(..., description="Default artifact root path.")
    terminate_apps_on_port: bool = Field(
        default=False, description="Whether to terminate apps on this port."
    )


class MLflowClientSettings(BaseSettings):
    experiment_name: str = Field(
        default="experiment", description="MLflow experiment name."
    )
    run_name: str = Field(None, description="Name of the MLflow run.")
    tracking_uri: str = Field(..., description="Tracking URI for MLflow.")
    register_model: bool = Field(
        default=False, description="Whether to register the model."
    )


class MLflowSettings(BaseSettings):
    server: MLflowServerSettings = Field(..., description="MLflow server settings.")
    client: MLflowClientSettings = Field(..., description="MLflow client settings.")
