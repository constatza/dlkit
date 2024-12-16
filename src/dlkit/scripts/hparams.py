from datetime import datetime

import optuna
import argparse
import mlflow
import torch
from pydantic import validate_call
from pydantic.types import FilePath
from lightning.pytorch import seed_everything

from dlkit.io.readers import load_config
from dlkit.optimization import objective
from dlkit.setup.pruner import initialize_pruner
from dlkit.io.logging import get_logger
from dlkit.setup.tracking import initialize_mlflow_client
from dlkit.setup.datamodule import initialize_datamodule
from dlkit.utils.system_utils import import_dynamically
from dlkit.io.writers import write_toml


logger = get_logger(__name__)

# set all seeds with pytorch lightning
seed_everything(1)


@validate_call
def optimize(config_path: FilePath):
    config = load_config(config_path)
    torch.set_float32_matmul_precision("medium")

    dataset_module = initialize_datamodule(config)
    dataset_module.prepare_data()

    # Read n_trials from config
    optuna_config = config.get("optuna", {})
    n_trials = optuna_config["n_trials"]

    # setup mlflow experiment and tracking uri
    experiment_id = initialize_mlflow_client(config)

    # Setup pruner
    pruner = initialize_pruner(config.get("pruner"))
    logger.info(f"Using pruner: {pruner.__class__.__name__}")

    with mlflow.start_run(
        run_name=str(config["mlflow"].get("run_name") + f"-{datetime.now()}")
    ) as parent_run:
        mlflow.pytorch.autolog(log_models=config["mlflow"].get("log_models", False))
        study = optuna.create_study(
            direction=config["optuna"].get("direction", "minimize"),
            pruner=pruner,
            study_name=f"study_{experiment_id}",
        )
        study.optimize(
            lambda trial: objective(trial, config, dataset_module),
            n_trials=n_trials,
        )

        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best parameters: {study.best_trial.params}")
        logger.info(f"Best value: {study.best_trial.value}")

        # log best parameters to mlflow
        config["model"].update(study.best_trial.params)

        model_class = import_dynamically(
            config["model"].get("name"), prepend="dlkit.networks"
        )
        best_run_id = study.best_trial.user_attrs.get("mlflow_run_id")
        config["mlflow"].update(
            {"best_run_id": best_run_id, "run_name": f"best-{best_run_id}"}
        )
        write_toml(config, config_path.with_name("best_config.toml"))
