# optimization/objective.py
from wrapt import when_imported

import optuna
from dlkit.setup.datamodule import initialize_datamodule
from dlkit.setup.model import initialize_model
from dlkit.setup.trainer import initialize_trainer
from dlkit.setup.trial import suggest
import mlflow


def objective(trial, config, dataset_module):

    trial_hparams = suggest(trial, config)
    trainer = initialize_trainer(config)
    model = initialize_model(trial_hparams, dataset_module.shapes)

    with mlflow.start_run(run_name=f"{trial.number}", nested=True):
        trial.set_user_attr("mlflow_run_id", mlflow.active_run().info.run_id)
        mlflow.log_params(trial.params)

        trainer.fit(model, datamodule=dataset_module)
        val_loss = trainer.callback_metrics.get("val_loss")
        trainer.test(model, datamodule=dataset_module)

        if val_loss is not None:
            trial.report(val_loss.item(), step=trainer.current_epoch)
            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            return val_loss
        else:
            raise ValueError("Validation loss not found in callback metrics.")
