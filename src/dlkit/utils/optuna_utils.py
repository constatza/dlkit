import mlflow
import optuna
from lightning.pytorch import LightningDataModule

from dlkit.run.mlflow_training import train_mlflow
from dlkit.settings import Settings
from loguru import logger


def objective_mlflow(trial, settings: Settings, datamodule: LightningDataModule) -> float:
    trial.set_user_attr("mlflow_run_id", mlflow.active_run().info.run_id)
    logger.info(f"Optuna trial number:{trial.number}")

    # Train the model and log metrics
    trial_model_settings = settings.MODEL.resolve(trial)
    training_state = train_mlflow(
        settings.model_copy(update={"MODEL": trial_model_settings}),
        datamodule=datamodule,
    )
    trainer = training_state.trainer
    test_loss = trainer.logged_metrics.get("test_loss")

    if test_loss is not None:
        trial.report(test_loss.item(), step=trainer.current_epoch)
        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return test_loss.item()
    else:
        raise ValueError("Validation loss not found in callback metrics.")
