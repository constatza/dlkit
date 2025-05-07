from pydantic import Field, ValidationInfo, field_validator

from .base_settings import BaseSettings


class MLflowServerSettings(BaseSettings):
	scheme: str = Field(default='http', description='MLflow server scheme.')
	host: str = Field(default='127.0.0.1', description='MLflow server host address.')
	port: int = Field(default=5000, description='MLflow server port number.')
	backend_store_uri: str = Field(
		default='./mlruns/mlflow.db', description='URI for the backend store.'
	)
	artifacts_destination: str = Field(
		default='./mlruns/artifacts', description='Default artifact root path.'
	)


class MLflowClientSettings(BaseSettings):
	experiment_name: str = Field(default='Experiment', description='MLflow experiment name.')
	run_name: str = Field(default='Run', description='Name of the MLflow run.')
	tracking_uri: str | None = Field(default=None, description='Tracking URI for MLflow.')
	register_model: bool = Field(default=False, description='Whether to register the model.')
	max_trials: int = Field(
		default=3, description='Maximum number of trials for reaching mlflow server.'
	)


class MLflowSettings(BaseSettings):
	server: MLflowServerSettings = Field(
		default=MLflowServerSettings(), description='MLflow server settings.'
	)
	client: MLflowClientSettings = Field(
		default=MLflowClientSettings(), description='MLflow client settings.'
	)

	@field_validator('client', mode='after', check_fields=True)
	@classmethod
	def default_tracking_uri(cls, value: MLflowClientSettings, info: ValidationInfo):
		if value.tracking_uri is None:
			scheme = info.data['server'].scheme
			host = info.data['server'].host
			port = info.data['server'].port
			return value.model_copy(update={'tracking_uri': f'{scheme}://{host}:{port}'})
		return value
