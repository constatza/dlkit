import sys
import traceback
import optuna
import mlflow
import torch
from lightning.pytorch import seed_everything

from dlkit.setup.pruner import initialize_pruner
from dlkit.io.logging import get_logger
from dlkit.setup.tracking import initialize_mlflow_client
from dlkit.setup.datamodule import initialize_datamodule
from dlkit.utils.system_utils import import_dynamically
from dlkit.utils.optuna_utils import objective
from dlkit.io.readers import parse_config
from dlkit.setup.tracking import MLFlowConfig


logger = get_logger(__name__)

# set all seeds with pytorch lightning
seed_everything(1)
torch.set_float32_matmul_precision("medium")


def main(config_dict: dict):
    mlflow_config = MLFlowConfig(**config_dict["mlflow"])

    dataset_module = initialize_datamodule(config_dict)
    dataset_module.prepare_data()

    # Read n_trials from config
    optuna_config = config_dict.get("optuna", {})
    n_trials = optuna_config["n_trials"]

    # setup mlflow experiment and tracking uri
    experiment_id = initialize_mlflow_client(mlflow_config)

    # Setup pruner
    pruner = initialize_pruner(config_dict.get("pruner"))
    logger.info(f"Using pruner: {pruner.__class__.__name__}")
    experiment_name = config_dict["mlflow"].get("experiment_name", "experiment")

    with mlflow.start_run() as parent_run:
        mlflow.pytorch.autolog(
            log_models=config_dict["mlflow"].get("log_models", False)
        )
        study = optuna.create_study(
            direction=config_dict["optuna"].get("direction", "minimize"),
            pruner=pruner,
            study_name=f"study_{experiment_id}",
        )
        study.optimize(
            lambda trial: objective(trial, config_dict, dataset_module),
            n_trials=n_trials,
        )

        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best parameters: {study.best_trial.params}")
        logger.info(f"Best value: {study.best_trial.value}")

        # log best parameters to mlflow
        config_dict["model"].update(study.best_trial.params)

        model_class = import_dynamically(
            config_dict["model"].get("name"), prepend="dlkit.networks"
        )
        best_run_id = study.best_trial.user_attrs.get("mlflow_run_id")
        config_dict["mlflow"].update(
            {"best_run_id": best_run_id, "run_name": f"best-{best_run_id}"}
        )

        mlflow.log_dict(config_dict, "best_config.toml")


if __name__ == "__main__":
    try:
        config = parse_config(
            description="Hyperparameter Optimization with Optuna script."
        )
        main(config)
    except Exception as e:

        logger.error(e)
        logger.error(traceback.format_exc())
    finally:
        sys.exit(0)
