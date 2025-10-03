"""Composable tracking decorator following OCP and SRP."""

from __future__ import annotations

from dlkit.interfaces.api.domain import TrainingResult
from dlkit.tools.config import GeneralSettings
from dlkit.runtime.workflows.factories.build_factory import BuildComponents
from dlkit.tools.utils.error_handling import raise_error
from dlkit.tools.utils.logging_config import get_logger

from ..core.interfaces import ITrainingExecutor
from .interfaces import IExperimentTracker, IRunContext

logger = get_logger(__name__)


class TrackingDecorator(ITrainingExecutor):
    """Composable experiment tracking decorator following OCP, DIP, and SRP.

    Adds tracking capability to any training executor without modifying it.
    Resource management is delegated to the tracker implementation.
    """

    def __init__(self, executor: ITrainingExecutor, tracker: IExperimentTracker):
        """Initialize with base executor and tracker.

        Args:
            executor: Core training executor to decorate
            tracker: Experiment tracker implementation
        """
        self._executor = executor
        self._tracker = tracker

    def execute(self, components: BuildComponents, settings: GeneralSettings) -> TrainingResult:
        """Execute training with experiment tracking.

        Args:
            components: Pre-built training components
            settings: Global training settings

        Returns:
            TrainingResult enriched with tracking metadata

        Raises:
            WorkflowError: If training or tracking fails
        """
        # Configure the tracker before entering its context so resources initialize correctly
        logger.debug("Setting up tracking")
        server_url, server_status = self._setup_tracking(settings)

        # Use tracker as context manager for proper resource management
        logger.info("Starting training with MLflow tracking")
        with self._tracker:
            logger.debug("Entering tracker context, resolving metadata")
            server_url, server_status = self._resolve_tracker_metadata(
                server_url, server_status
            )
            logger.debug("Executing training with tracking")
            result = self._execute_with_tracking(components, settings, server_url, server_status)
            logger.debug("Training completed, exiting tracker context")
            return result

    def _execute_with_tracking(
        self,
        components: BuildComponents,
        settings: GeneralSettings,
        server_url: str | None = None,
        server_status: dict | None = None,
    ) -> TrainingResult:
        """Execute training with tracking - separated for SRP compliance.

        Args:
            components: Pre-built training components
            settings: Global training settings

        Returns:
            TrainingResult enriched with tracking metadata
        """
        try:
            # Get run configuration
            logger.debug("Extracting run config")
            run_config = self._extract_run_config(settings)

            # Execute training within tracking context
            logger.debug(f"Creating MLflow run with config: {run_config}")
            with self._tracker.create_run(**run_config) as run_context:
                logger.debug("MLflow run created successfully")
                # Log metadata and configuration
                self._log_tracking_metadata(run_context, server_url, server_status)
                self._log_configuration(components, settings, run_context)

                # Inject MLflow logger into trainer for automatic metric logging
                self._inject_mlflow_logger(components, run_context, settings)

                # Execute core training
                result = self._executor.execute(components, settings)

                # Log resulting metrics before artifacts so MLflow captures them even if
                # artifact logging fails later on.
                self._log_training_metrics(result, run_context)

                # Log training artifacts
                self._log_training_artifacts(components, settings, run_context)

                # Log user-defined custom artifacts and params
                self._log_user_artifacts(components, settings, run_context, result)

                # Enrich result with tracking metadata
                return self._enrich_result(result, server_url, server_status)

        except Exception as e:
            logger.error(f"TrackingDecorator: Exception caught: {type(e).__name__}: {e}")
            raise_error("Training with tracking failed", e, stage="tracking")

    def _setup_tracking(self, settings: GeneralSettings) -> tuple[str | None, dict | None]:
        """Setup tracking configuration.

        Args:
            settings: Global settings

        Returns:
            Tuple of (server_url, server_status)
        """
        mlflow_config = settings.MLFLOW
        if hasattr(self._tracker, "setup_mlflow_config"):
            root_dir = None
            session = getattr(settings, "SESSION", None)
            if session is not None:
                root_dir = getattr(session, "root_dir", None)
            setup_fn = getattr(self._tracker, "setup_mlflow_config")
            try:
                return setup_fn(mlflow_config, root_dir=root_dir)
            except TypeError as exc:
                # Fall back for trackers that have not yet adopted the root_dir kwarg
                if "unexpected keyword argument 'root_dir'" in str(exc):
                    return setup_fn(mlflow_config)
                raise
        return None, None

    def _resolve_tracker_metadata(
        self,
        server_url: str | None,
        server_status: dict | None,
    ) -> tuple[str | None, dict | None]:
        """Merge tracker-provided metadata discovered during context entry."""
        # Prefer metadata returned from setup if provided; otherwise, ask tracker now
        if server_url is None and hasattr(self._tracker, "get_server_url"):
            try:
                candidate_url = self._tracker.get_server_url()
            except TypeError:
                candidate_url = self._tracker.get_server_url(server_url)
            if candidate_url:
                server_url = candidate_url

        if server_status is None and hasattr(self._tracker, "get_server_status"):
            try:
                candidate_status = self._tracker.get_server_status(server_url)
            except TypeError:
                candidate_status = self._tracker.get_server_status()
            if candidate_status is not None:
                server_status = candidate_status

        return server_url, server_status

    def _extract_run_config(self, settings: GeneralSettings) -> dict:
        """Extract run configuration from settings.

        Args:
            settings: Global settings

        Returns:
            Run configuration dictionary
        """
        from .naming import determine_experiment_name

        mlflow_config = settings.MLFLOW
        client = getattr(mlflow_config, "client", None) if mlflow_config else None

        return {
            "experiment_name": determine_experiment_name(settings, mlflow_config),
            "run_name": getattr(client, "run_name", None) if client else None,
            "nested": self._should_use_nested_runs(),
        }

    def _log_tracking_metadata(
        self,
        run_context: IRunContext,
        server_url: str | None,
        server_status: dict | None
    ) -> None:
        """Log tracking metadata to run context.

        Args:
            run_context: Run context for logging
            server_url: Server URL if available
            server_status: Server status if available
        """
        if server_url:
            run_context.set_tag("mlflow_server_url", server_url)

        if server_status is not None:
            run_context.set_tag(
                "mlflow_server_running", str(bool(server_status.get("running")))
            )
            run_context.set_tag(
                "mlflow_server_response_time", str(server_status.get("response_time"))
            )

    def _log_configuration(
        self,
        components: BuildComponents,
        settings: GeneralSettings,
        run_context: IRunContext
    ) -> None:
        """Log configuration and model parameters.

        Args:
            components: Build components
            settings: Global settings
            run_context: Run context for logging
        """
        # Log configuration
        self._tracker.log_settings(settings, run_context)

        # Log model hyperparameters
        self._tracker.log_model_parameters(components.model, run_context, settings)

    def _inject_mlflow_logger(
        self,
        components: BuildComponents,
        run_context: IRunContext,
        settings: GeneralSettings,
    ) -> None:
        """Inject MLflow epoch logger callback for metric logging with epochs as x-axis.

        Args:
            components: Build components
            run_context: Run context with client and run_id
            settings: Global settings
        """
        try:
            trainer = getattr(components, "trainer", None)
            if not trainer:
                return

            # Create callback that logs metrics with epoch numbers instead of steps
            from dlkit.core.training.callbacks import MLflowEpochLogger

            epoch_logger = MLflowEpochLogger(run_context)

            # Add callback to trainer
            if not hasattr(trainer, "callbacks"):
                trainer.callbacks = []
            trainer.callbacks.append(epoch_logger)
            logger.debug(f"Injected MLflowEpochLogger callback (run_id={run_context.run_id})")

        except Exception as e:
            logger.warning(f"Failed to inject MLflow epoch logger: {e}")

    def _log_training_artifacts(
        self,
        components: BuildComponents,
        settings: GeneralSettings,
        run_context: IRunContext
    ) -> None:
        """Log training artifacts.

        Args:
            components: Build components
            settings: Global settings
            run_context: Run context for logging
        """
        self._log_checkpoints(components, run_context)
        self._maybe_register_model(settings.MLFLOW, settings)

    def _log_training_metrics(
        self,
        result: TrainingResult,
        run_context: IRunContext,
    ) -> None:
        """Log scalar metrics from the training result to the tracker."""
        metrics = getattr(result, "metrics", None)
        if not metrics:
            return

        import math

        numeric: dict[str, float] = {}
        fallback: dict[str, str] = {}

        for key, value in metrics.items():
            try:
                numeric_value = float(value)
                if math.isnan(numeric_value) or math.isinf(numeric_value):
                    raise ValueError("non-finite metric")
                numeric[key] = numeric_value
            except (TypeError, ValueError):
                if value is not None:
                    fallback[f"metric_{key}"] = str(value)

        if numeric:
            run_context.log_metrics(numeric)
        if fallback:
            run_context.log_params(fallback)

    def _log_user_artifacts(
        self,
        components: BuildComponents,
        settings: GeneralSettings,
        run_context: IRunContext,
        result: TrainingResult,
    ) -> None:
        """Orchestrate logging of user-defined artifacts and params from settings.EXTRAS.

        Args:
            components: Build components
            settings: Global settings
            run_context: Run context for logging
            result: Training result
        """
        if not hasattr(settings, "EXTRAS") or settings.EXTRAS is None:
            return

        try:
            self._log_user_params(settings.EXTRAS, run_context)
            self._log_user_file_artifacts(settings.EXTRAS, run_context)
            self._log_user_toml_artifacts(settings.EXTRAS, run_context)
        except Exception as e:
            logger.warning(f"Failed to log user-defined artifacts/params: {e}")

    def _log_user_params(self, extras: Any, run_context: IRunContext) -> None:
        """Log user-defined parameters from EXTRAS.mlflow_params.

        Args:
            extras: EXTRAS settings object
            run_context: Run context for logging
        """
        if not hasattr(extras, "mlflow_params") or not extras.mlflow_params:
            return

        params_dict = extras.mlflow_params
        if not isinstance(params_dict, dict):
            return

        # Filter out non-serializable values
        safe_params = {}
        for key, value in params_dict.items():
            try:
                safe_params[key] = str(value) if value is not None else ""
            except Exception:
                logger.warning(f"Skipping non-serializable param: {key}")

        if safe_params:
            run_context.log_params(safe_params)
            logger.debug(f"Logged {len(safe_params)} custom params from EXTRAS.mlflow_params")

    def _log_user_file_artifacts(self, extras: Any, run_context: IRunContext) -> None:
        """Log user-defined file artifacts from EXTRAS.mlflow_artifacts.

        Args:
            extras: EXTRAS settings object
            run_context: Run context for logging
        """
        if not hasattr(extras, "mlflow_artifacts") or not extras.mlflow_artifacts:
            return

        from pathlib import Path

        artifacts = extras.mlflow_artifacts
        if not isinstance(artifacts, (list, tuple)):
            artifacts = [artifacts]

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
                logger.warning(f"Failed to log artifact {artifact_path}: {e}")

    def _log_user_toml_artifacts(self, extras: Any, run_context: IRunContext) -> None:
        """Log user-defined dicts as TOML artifacts from EXTRAS.mlflow_artifacts_toml.

        Converts dict values to TOML files and logs them to MLflow.

        Args:
            extras: EXTRAS settings object
            run_context: Run context for logging
        """
        if not hasattr(extras, "mlflow_artifacts_toml") or not extras.mlflow_artifacts_toml:
            return

        from dlkit.tools.io.config import write_config
        from pathlib import Path
        import tempfile
        import os

        artifacts_toml = extras.mlflow_artifacts_toml
        if not isinstance(artifacts_toml, dict):
            return

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
                logger.warning(f"Failed to log TOML artifact {name}: {e}")

    def _log_checkpoints(self, components: BuildComponents, run_context: IRunContext) -> None:
        """Log model checkpoints as artifacts."""
        try:
            trainer = getattr(components, "trainer", None)
            if not trainer:
                return

            # Find ModelCheckpoint callback
            ckpt_cb = getattr(trainer, "checkpoint_callback", None)
            if not ckpt_cb and hasattr(trainer, "callbacks"):
                try:
                    from pytorch_lightning.callbacks import ModelCheckpoint

                    cands = [c for c in trainer.callbacks if isinstance(c, ModelCheckpoint)]
                    ckpt_cb = cands[0] if cands else None
                except ImportError:
                    return

            if not ckpt_cb:
                return

            # Log best and last checkpoints
            best = getattr(ckpt_cb, "best_model_path", None)
            last = getattr(ckpt_cb, "last_model_path", None)

            if best:
                from pathlib import Path

                run_context.log_artifact(Path(best), "checkpoints")
            if last and last != best:
                from pathlib import Path

                run_context.log_artifact(Path(last), "checkpoints")

        except Exception:
            pass

    def _maybe_register_model(self, mlflow_config, settings: GeneralSettings) -> None:
        """Register model in MLflow model registry if configured."""
        try:
            client = getattr(mlflow_config, "client", None) if mlflow_config else None
            if not (client and getattr(client, "register_model", False)):
                return

            # Use client-based approach if available
            if hasattr(self._tracker, '_resource_manager') and self._tracker._resource_manager:
                client_instance = self._tracker._resource_manager.get_client()
                # TODO: Implement client-based model registration
                logger.debug("Client-based model registration not yet implemented")
                return

            # Fallback to global MLflow for now
            import mlflow

            # Log model artifact
            mlflow.pytorch.log_model(settings.components.model, artifact_path="model")

            # Register model
            run = mlflow.active_run()
            if not run:
                return

            model_name = str(getattr(getattr(settings, "MODEL", None), "name", "DLKitModel"))
            mlflow.register_model(model_uri=f"runs:/{run.info.run_id}/model", name=model_name)
        except Exception:
            pass

    def _should_use_nested_runs(self) -> bool:
        """Determine if nested runs should be used based on existing MLflow run context."""
        try:
            import mlflow

            # Check if there's already an active MLflow run
            active_run = mlflow.active_run()
            return active_run is not None
        except Exception:
            return False

    def _enrich_result(
        self,
        result: TrainingResult,
        server_url: str | None,
        server_status: dict | None,
    ) -> TrainingResult:
        """Enrich training result with tracking metadata."""
        if not isinstance(result, TrainingResult):
            return result

        try:
            enriched = dict(getattr(result, "metrics", {}) or {})

            # Add MLflow run information using client if available
            if hasattr(self._tracker, '_resource_manager') and self._tracker._resource_manager:
                try:
                    # Use client-based approach for getting run info
                    # This avoids dependency on global MLflow state
                    logger.debug("Client-based result enrichment not yet fully implemented")
                except Exception:
                    pass

            # Fallback to global MLflow for now
            try:
                import mlflow

                run = mlflow.active_run()
                if run:
                    enriched["mlflow_run_id"] = run.info.run_id
                    enriched["mlflow_experiment_id"] = run.info.experiment_id

                # Use server URL as tracking URI if available (we use MlflowClient with explicit URI)
                # Otherwise fall back to client's tracking URI
                if server_url:
                    enriched["mlflow_tracking_uri"] = server_url
                else:
                    try:
                        client = self._tracker.get_client()
                        enriched["mlflow_tracking_uri"] = getattr(client, "tracking_uri", None)
                    except Exception:
                        pass
            except Exception:
                pass

            if server_url:
                enriched["mlflow_server_url"] = server_url
            if server_status is not None:
                enriched["mlflow_server_running"] = bool(server_status.get("running"))
                if server_status.get("response_time") is not None:
                    enriched["mlflow_server_response_time"] = server_status["response_time"]

            return TrainingResult(
                model_state=result.model_state,
                metrics=enriched,
                artifacts=result.artifacts,
                duration_seconds=result.duration_seconds,
            )
        except Exception:
            return result
