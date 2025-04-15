import mlflow
import optuna

from dlkit.datamodules import NumpyModule
from dlkit.settings import ModelSettings, TrainerSettings
from dlkit.setup.model import initialize_model
from dlkit.setup.trainer import initialize_trainer


# TODO: use train
def objective(
    trial,
    model_settings: ModelSettings,
    datamodule: NumpyModule,
    trainer_settings: TrainerSettings,
) -> float:

    model = initialize_model(model_settings.resolve(trial), datamodule.shape)
    trainer_settings = initialize_trainer(trainer_settings)

    with mlflow.start_run(
        run_name=f"{trial.number}",
        nested=True,
        experiment_id=mlflow.active_run().info.experiment_id,
    ):
        trial.set_user_attr("mlflow_run_id", mlflow.active_run().info.run_id)
        mlflow.log_params(trial.params)

        trainer_settings.fit(model, datamodule=datamodule)
        val_loss = trainer_settings.callback_metrics.get("val_loss")
        trainer_settings.test(model, datamodule=datamodule)

        if val_loss is not None:
            trial.report(val_loss.item(), step=trainer_settings.current_epoch)
            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            return val_loss
        else:
            raise ValueError("Validation loss not found in callback metrics.")
