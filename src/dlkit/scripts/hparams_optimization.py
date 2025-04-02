import sys
import traceback
import optuna
import mlflow
import torch
from lightning.pytorch import seed_everything
from loguru import logger
from pydantic import validate_call, FilePath
import click

from dlkit.io.settings import load_validated_settings
from dlkit.setup.pruner import initialize_pruner
from dlkit.setup.tracking import initialize_mlflow_client
from dlkit.setup.datamodule import initialize_datamodule
from dlkit.setup.trainer import initialize_trainer
from dlkit.utils.optuna_utils import objective


# set all seeds with pytorch lightning
seed_everything(1)
torch.set_float32_matmul_precision("medium")


@click.command(
    "Hyperparameter Optimization", help="Hyperparameter Optimization with Optuna."
)
@click.argument("config-path")
@validate_call
def main(config_path: FilePath) -> None:
    settings = load_validated_settings(config_path)

    datamodule = initialize_datamodule(settings.DATAMODULE, settings.PATHS)
    datamodule.setup(stage="fit")

    # setup mlflow experiment and tracking uri
    experiment_id = initialize_mlflow_client(settings.MLFLOW.client)

    # Setup pruner
    pruner = initialize_pruner(settings.OPTUNA.pruner)

    with mlflow.start_run(
        experiment_id=experiment_id, run_name=settings.MLFLOW.client.run_name
    ) as parent_run:
        mlflow.pytorch.autolog(log_models=False)
        study = optuna.create_study(
            direction=settings.OPTUNA.direction,
            pruner=pruner,
            study_name=f"study_{experiment_id}",
        )
        study.optimize(
            lambda trial: objective(
                trial, settings.MODEL, datamodule, settings.TRAINER
            ),
            n_trials=settings.OPTUNA.n_trials,
        )

        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best parameters: {study.best_trial.params}")
        logger.info(f"Best value: {study.best_trial.value}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(traceback.format_exc())
    finally:
        sys.exit(0)
