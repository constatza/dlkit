"""Override manager for applying runtime parameter overrides to GeneralSettings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dlkit.tools.config import GeneralSettings


class BasicOverrideManager:
    """Manager for applying basic runtime overrides to GeneralSettings.

    This class handles the proper application of runtime parameter overrides
    to Pydantic GeneralSettings objects using model_copy() to maintain
    immutability and type safety.
    """

    def apply_overrides(
        self,
        base_settings: GeneralSettings,
        **overrides: Any,
    ) -> GeneralSettings:
        """Apply runtime overrides to base settings using Pydantic model_copy().

        Args:
            base_settings: Base GeneralSettings object
            **overrides: Runtime parameter overrides

        Returns:
            New GeneralSettings object with overrides applied

        Example:
            >>> manager = BasicOverrideManager()
            >>> new_settings = manager.apply_overrides(
            ...     settings,
            ...     checkpoint_path=Path("./model.ckpt"),
            ...     output_dir=Path("./output"),
            ...     epochs=100,
            ...     mlflow_host="localhost",
            ... )
        """
        current_settings = base_settings

        # Apply path overrides via environment variables
        current_settings = self._apply_environment_path_overrides(current_settings, overrides)

        # Apply model/checkpoint overrides
        current_settings = self._apply_model_overrides(current_settings, overrides)

        # Apply training overrides
        current_settings = self._apply_training_overrides(current_settings, overrides)

        # Apply MLflow overrides
        current_settings = self._apply_mlflow_overrides(current_settings, overrides)

        # Apply Optuna overrides
        current_settings = self._apply_optuna_overrides(current_settings, overrides)

        return current_settings

    def _apply_environment_path_overrides(
        self,
        settings: GeneralSettings,
        overrides: dict[str, Any],
    ) -> GeneralSettings:
        """Apply path-related overrides via path context.

        This method sets up a path override context that will be used by
        domain functions when resolving component paths. This maintains
        clean separation between API overrides and environment configuration.

        Args:
            settings: Current settings
            overrides: Dictionary containing potential path overrides

        Returns:
            GeneralSettings: Settings object (unchanged, context set for thread)
        """
        from dlkit.interfaces.api.overrides.path_context import (
            set_path_context,
            PathOverrideContext,
        )

        # Collect path overrides
        context_overrides = {}
        if "root_dir" in overrides and overrides["root_dir"] is not None:
            context_overrides["root_dir"] = overrides["root_dir"]
        if "output_dir" in overrides and overrides["output_dir"] is not None:
            context_overrides["output_dir"] = overrides["output_dir"]
        if "data_dir" in overrides and overrides["data_dir"] is not None:
            context_overrides["data_dir"] = overrides["data_dir"]
        if "checkpoints_dir" in overrides and overrides["checkpoints_dir"] is not None:
            context_overrides["checkpoints_dir"] = overrides["checkpoints_dir"]

        # Set the path context for this thread if we have overrides
        if context_overrides:
            context = PathOverrideContext(
                root_dir=Path(context_overrides["root_dir"])
                if context_overrides.get("root_dir")
                else None,
                output_dir=Path(context_overrides["output_dir"])
                if context_overrides.get("output_dir")
                else None,
                data_dir=Path(context_overrides["data_dir"])
                if context_overrides.get("data_dir")
                else None,
                checkpoints_dir=Path(context_overrides["checkpoints_dir"])
                if context_overrides.get("checkpoints_dir")
                else None,
            )
            set_path_context(context)

        return settings

    def _apply_model_overrides(
        self,
        settings: GeneralSettings,
        overrides: dict[str, Any],
    ) -> GeneralSettings:
        """Apply model and checkpoint overrides."""
        if "checkpoint_path" not in overrides or overrides["checkpoint_path"] is None:
            return settings

        # Update top-level MODEL checkpoint (shallow hierarchy)
        if settings.MODEL is None:
            return settings

        new_model = settings.MODEL.model_copy(update={"checkpoint": overrides["checkpoint_path"]})
        return settings.model_copy(update={"MODEL": new_model})

    def _apply_training_overrides(
        self,
        settings: GeneralSettings,
        overrides: dict[str, Any],
    ) -> GeneralSettings:
        """Apply training-related overrides using flattened structure."""
        training_overrides = {
            k: v
            for k, v in overrides.items()
            if k in ["epochs", "batch_size", "learning_rate", "train", "test", "predict"]
            and v is not None
        }

        if not training_overrides or not settings.TRAINING:
            return settings

        current_settings = settings

        # Handle trainer overrides (epochs) - use flattened TRAINING
        if "epochs" in training_overrides:
            trainer_overrides = {"max_epochs": training_overrides["epochs"]}
            new_trainer = settings.TRAINING.trainer.model_copy(update=trainer_overrides)
            # Update both TRAINING.epochs and TRAINING.trainer.max_epochs to keep in sync
            new_training = settings.TRAINING.model_copy(
                update={"trainer": new_trainer, "epochs": training_overrides["epochs"]}
            )
            current_settings = current_settings.model_copy(update={"TRAINING": new_training})

        # (Removed) pipeline flag overrides; not part of simplified TrainingSettings

        # Handle batch_size override -> DATAMODULE.batch_size and DATAMODULE.dataloader.batch_size
        if "batch_size" in training_overrides and current_settings.DATAMODULE is not None:
            dm = current_settings.DATAMODULE
            # Update convenience field
            dm_updates: dict[str, Any] = {"batch_size": training_overrides["batch_size"]}
            # Also update nested dataloader setting to ensure effective config everywhere
            if hasattr(dm, "dataloader") and dm.dataloader is not None:
                new_loader = dm.dataloader.model_copy(
                    update={"batch_size": training_overrides["batch_size"]}
                )
                dm_updates["dataloader"] = new_loader
            new_dm = dm.model_copy(update=dm_updates)
            current_settings = current_settings.model_copy(update={"DATAMODULE": new_dm})

        # Handle learning_rate override -> TRAINING.optimizer.lr
        if "learning_rate" in training_overrides and current_settings.TRAINING is not None:
            tr = current_settings.TRAINING
            opt_updates = {"lr": float(training_overrides["learning_rate"])}
            new_opt = tr.optimizer.model_copy(update=opt_updates)
            new_training = tr.model_copy(update={"optimizer": new_opt})
            current_settings = current_settings.model_copy(update={"TRAINING": new_training})

        return current_settings

    def _apply_mlflow_overrides(
        self,
        settings: GeneralSettings,
        overrides: dict[str, Any],
    ) -> GeneralSettings:
        """Apply MLflow-related overrides using flattened structure."""
        mlflow_overrides = {
            k: v
            for k, v in overrides.items()
            if (
                k.startswith("mlflow_")
                or k in ["experiment_name", "run_name", "enable_mlflow", "tracking_uri", "backend_store_uri", "artifacts_destination"]
            )
            and v is not None
        }

        if not mlflow_overrides:
            return settings

        # Handle enabling MLflow when not present
        if (
            "enable_mlflow" in mlflow_overrides
            and mlflow_overrides["enable_mlflow"]
            and not settings.MLFLOW
        ):
            from dlkit.tools.config.mlflow_settings import MLflowSettings, MLflowClientSettings

            # Create minimal MLflow settings - user should configure properly
            experiment_name = mlflow_overrides.get("experiment_name", "default_experiment")
            tracking_uri = mlflow_overrides.get("tracking_uri", "http://localhost:5000")

            default_mlflow = MLflowSettings(
                enabled=True,
                client=MLflowClientSettings(
                    experiment_name=experiment_name, tracking_uri=tracking_uri
                ),
            )
            current_mlflow = default_mlflow
            settings = settings.model_copy(update={"MLFLOW": current_mlflow})
        elif not settings.MLFLOW:
            return settings
        else:
            current_mlflow = settings.MLFLOW

        # Apply server overrides
        server_overrides = {}
        if "mlflow_host" in mlflow_overrides:
            server_overrides["host"] = mlflow_overrides["mlflow_host"]
        if "mlflow_port" in mlflow_overrides:
            server_overrides["port"] = mlflow_overrides["mlflow_port"]
        if "backend_store_uri" in mlflow_overrides:
            server_overrides["backend_store_uri"] = mlflow_overrides["backend_store_uri"]
        if "artifacts_destination" in mlflow_overrides:
            server_overrides["artifacts_destination"] = mlflow_overrides["artifacts_destination"]

        if server_overrides:
            new_server = current_mlflow.server.model_copy(update=server_overrides)
            current_mlflow = current_mlflow.model_copy(update={"server": new_server})

        # Apply client overrides
        client_overrides = {}
        if "experiment_name" in mlflow_overrides:
            client_overrides["experiment_name"] = mlflow_overrides["experiment_name"]
        if "run_name" in mlflow_overrides:
            client_overrides["run_name"] = mlflow_overrides["run_name"]
        if "tracking_uri" in mlflow_overrides:
            client_overrides["tracking_uri"] = mlflow_overrides["tracking_uri"]

        if client_overrides:
            new_client = current_mlflow.client.model_copy(update=client_overrides)
            current_mlflow = current_mlflow.model_copy(update={"client": new_client})

        # Apply to flattened structure
        return settings.model_copy(update={"MLFLOW": current_mlflow})

    def _apply_optuna_overrides(
        self,
        settings: GeneralSettings,
        overrides: dict[str, Any],
    ) -> GeneralSettings:
        """Apply Optuna-related overrides using flattened structure."""
        optuna_overrides = {
            k: v
            for k, v in overrides.items()
            if k in ["trials", "study_name", "enable_optuna"] and v is not None
        }

        if not optuna_overrides:
            return settings

        # Handle enabling Optuna when not present
        if (
            "enable_optuna" in optuna_overrides
            and optuna_overrides["enable_optuna"]
            and not settings.OPTUNA
        ):
            from dlkit.tools.config.optuna_settings import OptunaSettings

            # Create minimal Optuna settings - user should configure properly
            n_trials = optuna_overrides.get("trials", 3)  # Minimal default for testing
            study_name = optuna_overrides.get("study_name", "default_study")
            from dlkit.interfaces.servers.domain_functions import get_default_optuna_storage_url

            storage = optuna_overrides.get("storage", get_default_optuna_storage_url())

            default_optuna = OptunaSettings(
                enabled=True, n_trials=n_trials, study_name=study_name, storage=storage
            )
            current_optuna = default_optuna
            settings = settings.model_copy(update={"OPTUNA": current_optuna})
        elif not settings.OPTUNA:
            return settings
        else:
            current_optuna = settings.OPTUNA

        # Apply Optuna-specific overrides
        plugin_overrides = {}
        if "trials" in optuna_overrides:
            plugin_overrides["n_trials"] = optuna_overrides["trials"]
        if "study_name" in optuna_overrides:
            plugin_overrides["study_name"] = optuna_overrides["study_name"]
        if "enable_optuna" in optuna_overrides:
            plugin_overrides["enabled"] = optuna_overrides["enable_optuna"]

        if plugin_overrides:
            new_optuna = current_optuna.model_copy(update=plugin_overrides)
            return settings.model_copy(update={"OPTUNA": new_optuna})

        return settings

    def validate_overrides(
        self,
        settings: GeneralSettings,
        **overrides: Any,
    ) -> list[str]:
        """Validate runtime overrides against base settings.

        Args:
            settings: Base settings to validate against
            **overrides: Override parameters to validate

        Returns:
            List of validation error messages (empty if all valid)
        """
        errors = []

        # Validate checkpoint path exists if provided (ignore None values)
        if "checkpoint_path" in overrides and overrides["checkpoint_path"] is not None:
            checkpoint_path = overrides["checkpoint_path"]
            if not isinstance(checkpoint_path, Path):
                try:
                    checkpoint_path = Path(checkpoint_path)
                except (TypeError, ValueError):
                    errors.append("checkpoint_path must be a valid path")
                    return errors

            if not checkpoint_path.exists():
                errors.append(f"Checkpoint file does not exist: {checkpoint_path}")

        # Validate numeric parameters (ignore None values)
        numeric_params = ["epochs", "batch_size", "learning_rate", "mlflow_port", "trials"]
        for param in numeric_params:
            if param in overrides and overrides[param] is not None:
                value = overrides[param]
                if not isinstance(value, (int, float)) or value <= 0:
                    errors.append(f"{param} must be a positive number, got: {value}")

        # Validate MLflow overrides require MLflow plugin (ignore None values)
        mlflow_overrides = [
            k
            for k in overrides.keys()
            if (k.startswith("mlflow_") or k in ["experiment_name", "run_name"])
            and overrides[k] is not None
        ]
        if mlflow_overrides:
            if not (settings.MLFLOW and settings.MLFLOW.enabled):
                errors.append("MLflow overrides require MLflow to be enabled in configuration")

        # Validate Optuna overrides require Optuna plugin (ignore None values)
        optuna_overrides = [
            k
            for k in overrides.keys()
            if k in ["trials", "study_name"] and overrides[k] is not None
        ]
        if optuna_overrides:
            if not (settings.OPTUNA and settings.OPTUNA.enabled):
                errors.append("Optuna overrides require Optuna to be enabled in configuration")

        return errors


# Global override manager instance
basic_override_manager = BasicOverrideManager()
