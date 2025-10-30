"""Artifact logging service for MLflow tracking.

Single Responsibility: Log checkpoints, models, and user-defined artifacts to MLflow.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from dlkit.interfaces.api.domain import TrainingResult
from dlkit.runtime.workflows.factories.build_factory import BuildComponents
from dlkit.tools.config import GeneralSettings
from dlkit.tools.utils.logging_config import get_logger

from .config_accessor import ConfigAccessor
from .interfaces import IExperimentTracker, IRunContext

logger = get_logger(__name__)


class ArtifactLogger:
    """Handles artifact logging to MLflow.

    Single Responsibility: Log checkpoints, models, and user-defined artifacts.
    Delegates configuration access to ConfigAccessor.

    Args:
        tracker: Experiment tracker implementation
    """

    def __init__(self, tracker: IExperimentTracker):
        """Initialize with experiment tracker.

        Args:
            tracker: Experiment tracker implementation
        """
        self._tracker = tracker

    def log_training_artifacts(
        self,
        components: BuildComponents,
        settings: GeneralSettings,
        run_context: IRunContext,
    ) -> None:
        """Log all training artifacts (checkpoints and optional model registration).

        Args:
            components: Build components containing trainer
            settings: Global settings
            run_context: Run context for logging
        """
        self.log_checkpoints(components, run_context)
        self.maybe_register_model(settings, components, run_context)

    def log_checkpoints(
        self,
        components: BuildComponents,
        run_context: IRunContext,
    ) -> None:
        """Log model checkpoints as artifacts.

        Logs best and last checkpoints from PyTorch Lightning trainer.

        Args:
            components: Build components containing trainer
            run_context: Run context for logging
        """
        try:
            trainer = getattr(components, "trainer", None)
            if not trainer:
                logger.debug("No trainer found in components")
                return

            # Find ModelCheckpoint callback
            ckpt_cb = self._find_checkpoint_callback(trainer)
            if not ckpt_cb:
                logger.debug("No ModelCheckpoint callback found")
                return

            # Log best and last checkpoints
            best = getattr(ckpt_cb, "best_model_path", None)
            last = getattr(ckpt_cb, "last_model_path", None)

            if best:
                run_context.log_artifact(Path(best), "checkpoints")
                logger.debug(f"Logged best checkpoint: {best}")
            if last and last != best:
                run_context.log_artifact(Path(last), "checkpoints")
                logger.debug(f"Logged last checkpoint: {last}")

        except Exception as e:
            logger.warning(f"Failed to log checkpoints: {e}")

    def maybe_register_model(
        self,
        settings: GeneralSettings,
        components: BuildComponents,
        run_context: IRunContext,
    ) -> None:
        """Register model in MLflow model registry if configured.

        Uses global MLflow for model registration. Client-based registration
        requires MLflowClient.log_model which is not yet implemented.

        Args:
            settings: Global settings
            components: Build components containing the model
            run_context: Run context for logging
        """
        try:
            accessor = ConfigAccessor(settings)
            if not accessor.should_register_model():
                logger.debug("Model registration disabled")
                return

            # Client-based registration requires implementing MLflowClient.log_model
            # For now, use global MLflow approach which is simpler and well-tested
            logger.debug("Using global MLflow for model registration")
            self._register_model_global(settings, components)

        except Exception as e:
            logger.warning(f"Failed to register model: {e}")

    def log_user_artifacts(
        self,
        settings: GeneralSettings,
        run_context: IRunContext,
        result: TrainingResult,
    ) -> None:
        """Orchestrate logging of user-defined artifacts and params from settings.EXTRAS.

        Args:
            settings: Global settings
            run_context: Run context for logging
            result: Training result (not currently used but kept for extensibility)
        """
        accessor = ConfigAccessor(settings)
        extras = accessor.get_extras()
        if not extras:
            logger.debug("No EXTRAS configuration found")
            return

        try:
            self._log_user_params(accessor, run_context)
            self._log_user_file_artifacts(accessor, run_context)
            self._log_user_toml_artifacts(accessor, run_context)
        except Exception as e:
            logger.warning(f"Failed to log user-defined artifacts/params: {e}")

    def _find_checkpoint_callback(self, trainer: Any) -> Any | None:
        """Find ModelCheckpoint callback in trainer.

        Args:
            trainer: PyTorch Lightning trainer

        Returns:
            ModelCheckpoint callback or None
        """
        # Check direct attribute first
        ckpt_cb = getattr(trainer, "checkpoint_callback", None)
        if ckpt_cb:
            return ckpt_cb

        # Search in callbacks list
        if hasattr(trainer, "callbacks"):
            try:
                from pytorch_lightning.callbacks import ModelCheckpoint

                candidates = [c for c in trainer.callbacks if isinstance(c, ModelCheckpoint)]
                return candidates[0] if candidates else None
            except ImportError as e:
                logger.warning(f"Failed to import ModelCheckpoint: {e}")
                return None

        return None

    def _register_model_global(
        self,
        settings: GeneralSettings,
        components: BuildComponents,
    ) -> None:
        """Register model using global MLflow.

        Args:
            settings: Global settings containing model configuration
            components: Build components containing the model
        """
        import mlflow

        # Log model artifact
        mlflow.pytorch.log_model(components.model, artifact_path="model")  # type: ignore[attr-defined]
        logger.debug("Logged model artifact to MLflow")

        # Register model
        run = mlflow.active_run()
        if not run:
            logger.warning("No active MLflow run for model registration")
            return

        accessor = ConfigAccessor(settings)
        model_name = accessor.get_model_name()
        mlflow.register_model(model_uri=f"runs:/{run.info.run_id}/model", name=model_name)
        logger.info(f"Registered model '{model_name}' in MLflow registry")

    def _log_user_params(
        self,
        accessor: ConfigAccessor,
        run_context: IRunContext,
    ) -> None:
        """Log user-defined parameters from EXTRAS.mlflow_params.

        Args:
            accessor: Configuration accessor
            run_context: Run context for logging
        """
        params_dict = accessor.get_mlflow_params()
        if not params_dict:
            return

        # Filter out non-serializable values
        safe_params = {}
        for key, value in params_dict.items():
            try:
                safe_params[key] = str(value) if value is not None else ""
            except Exception as e:
                logger.warning(f"Skipping non-serializable param '{key}': {e}")

        if safe_params:
            run_context.log_params(safe_params)
            logger.debug(f"Logged {len(safe_params)} custom params from EXTRAS.mlflow_params")

    def _log_user_file_artifacts(
        self,
        accessor: ConfigAccessor,
        run_context: IRunContext,
    ) -> None:
        """Log user-defined file artifacts from EXTRAS.mlflow_artifacts.

        Args:
            accessor: Configuration accessor
            run_context: Run context for logging
        """
        artifacts = accessor.get_mlflow_artifacts()
        if not artifacts:
            return

        for artifact_path in artifacts:
            try:
                path = Path(artifact_path)
                if path.exists() and path.is_file():
                    artifact_dir = str(path.parent) if path.parent != Path(".") else ""
                    run_context.log_artifact(path, artifact_dir=artifact_dir)
                    logger.debug(f"Logged artifact: {artifact_path}")
                else:
                    logger.warning(f"Artifact not found or not a file: {artifact_path}")
            except Exception as e:
                logger.warning(f"Failed to log artifact '{artifact_path}': {e}")

    def _log_user_toml_artifacts(
        self,
        accessor: ConfigAccessor,
        run_context: IRunContext,
    ) -> None:
        """Log user-defined dicts as TOML artifacts from EXTRAS.mlflow_artifacts_toml.

        Converts dict values to TOML files and logs them to MLflow.

        Args:
            accessor: Configuration accessor
            run_context: Run context for logging
        """
        artifacts_toml = accessor.get_mlflow_artifacts_toml()
        if not artifacts_toml:
            return

        from dlkit.tools.io.config import write_config

        for name, data_dict in artifacts_toml.items():
            try:
                if not isinstance(data_dict, dict):
                    logger.warning(f"Skipping non-dict TOML artifact: {name}")
                    continue

                # Create temporary TOML file
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".toml", delete=False, prefix=f"{name}_"
                ) as temp_file:
                    temp_path = Path(temp_file.name)

                # Write dict to TOML
                write_config(data_dict, temp_path, exclude_none=True)

                # Log the TOML file
                run_context.log_artifact(temp_path, artifact_dir="config")

                # Clean up temp file
                os.unlink(temp_path)

                logger.debug(f"Logged TOML artifact: {name}.toml → config/")

            except Exception as e:
                logger.warning(f"Failed to log TOML artifact '{name}': {e}")
