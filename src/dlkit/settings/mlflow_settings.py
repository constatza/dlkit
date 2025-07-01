import os
from pydantic import Field, ValidationInfo, field_validator
from pydantic.networks import AnyUrl, FileUrl

from .base_settings import BasicSettings
from dlkit.utils.system_utils import recommended_gunicorn_workers


class MLflowServerSettings(BasicSettings):
    scheme: str = Field(default="http", description="MLflow server scheme.")
    host: str = Field(default="127.0.0.1", description="MLflow server host address.")
    port: int = Field(default=5000, description="MLflow server port number.", gt=0, lt=65536)
    backend_store_uri: AnyUrl | None = Field(default=None, description="URI for the backend store.")
    artifacts_destination: FileUrl | None = Field(
        default=None, description="Default artifact root path."
    )
    num_workers: int = Field(default=4, description="Number of workers.")
    keep_alive_interval: int = Field(
        default=5, description="Duration in seconds to keep the server alive, after inactivity."
    )
    shutdown_timeout: int = Field(default=5, description="Timeout in seconds for server shutdown.")

    @property
    def command(self) -> list[str]:
        workers = recommended_gunicorn_workers()
        command = [
            "mlflow",
            "server",
            "--host",
            str(self.host),
            "--port",
            str(self.port),
            "--backend-store-uri",
            str(self.backend_store_uri),
            "--artifacts-destination",
            str(self.artifacts_destination),
        ]
        if os.name != "nt":
            command.extend([
                "--gunicorn-opts",
                f"-w={workers}",
            ])
        else:
            command.extend([
                "--waitress-opts",
                f"--threads={workers}",
            ])

        return command


class MLflowClientSettings(BasicSettings):
    experiment_name: str = Field(default="Experiment", description="MLflow experiment name.")
    run_name: str | None = Field(default=None, description="Name of the MLflow run.")
    tracking_uri: AnyUrl | None = Field(default=None, description="Tracking URI for MLflow.")
    register_model: bool = Field(default=False, description="Whether to register the model.")
    max_trials: int = Field(
        default=3, description="Maximum number of trials for reaching mlflow server."
    )


class MLflowSettings(BasicSettings):
    enable: bool = Field(default=False, description="Whether to enable MLflow.")
    server: MLflowServerSettings = Field(
        default_factory=MLflowServerSettings, description="MLflow server settings."
    )
    client: MLflowClientSettings = Field(
        default_factory=MLflowClientSettings, description="MLflow client settings."
    )

    @field_validator("client", mode="after", check_fields=True)
    @classmethod
    def default_tracking_uri(cls, value: MLflowClientSettings, info: ValidationInfo):
        if value.tracking_uri is None:
            scheme = info.data["server"].scheme
            host = info.data["server"].host
            port = info.data["server"].port
            return value.model_copy(update={"tracking_uri": AnyUrl(f"{scheme}://{host}:{port}")})
        return value
