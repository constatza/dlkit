import mlflow
import optuna
from loguru import logger
from pydantic import validate_call
from functools import partial

from dlkit.setup.mlflow_server import ServerProcess
from dlkit.settings import Settings
from dlkit.setup.datamodule import build_datamodule
from dlkit.setup.mlflow_client import initialize_mlflow_client
from dlkit.utils.loading import init_class
from dlkit.utils.optuna_utils import objective_mlflow


@validate_call
def train_optuna(settings: Settings) -> None:
    """Runs an Optuna optimization process for hyperparameter tuning.
    This function initializes the MLflow client, builds the datamodule,
    and sets up the Optuna study. It then optimizes the objective function
    using the specified number of trials and logs the results to MLflow.
    Args:
        settings (Settings): The DLkit settings object containing configuration
            for the Optuna optimization process.
    """

    datamodule = build_datamodule(
        settings.DATAMODULE,
        settings.DATASET,
        settings.DATALOADER,
        settings.PATHS,
    )
    pruner = init_class(settings.OPTUNA.pruner)

    with ServerProcess(settings.MLFLOW.server):
        experiment_id = initialize_mlflow_client(settings.MLFLOW.client)

        logger.info(f"Starting Optuna optimization with experiment ID: {experiment_id}")
        with mlflow.start_run(
            experiment_id=experiment_id, run_name=settings.MLFLOW.client.run_name
        ):
            mlflow.pytorch.autolog()
            study = optuna.create_study(
                direction=settings.OPTUNA.direction,
                pruner=pruner,
                study_name=f"study_{experiment_id}",
            )
            obj_function = partial(objective_mlflow, settings=settings, datamodule=datamodule)
            study.optimize(
                obj_function,
                n_trials=settings.OPTUNA.n_trials,
            )

            logger.info(f"Best trial: {study.best_trial.number}")
            logger.info(f"Best parameters: {study.best_trial.params}")
            logger.info(f"Best value: {study.best_trial.value}")
