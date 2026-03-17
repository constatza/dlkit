"""MLflow adapter implementing tracking abstractions."""

from contextlib import contextmanager, ExitStack, AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


from dlkit.interfaces.api.overrides.path_context import (
    get_current_path_context,
    path_override_context,
)
from dlkit.tools.config import GeneralSettings
from dlkit.tools.utils.logging_config import get_logger

from .interfaces import IExperimentTracker, IRunContext
from .dataset_lineage import DatasetSourceCollector, StructuredDatasetLogger
from .mlflow_resource_manager import MLflowResourceManager

logger = get_logger(__name__)

# MLflow integration constants
MLFLOW_DEFAULT_EXPERIMENT = "DLKit"


@dataclass(frozen=True, slots=True, kw_only=True)
class TrackingSetupResult:
    """Typed result returned by :meth:`MLflowTracker.setup_mlflow_config`.

    Attributes:
        tracking_uri: Resolved MLflow tracking URI, or ``None`` when tracking is disabled.
        resolved_artifact_uri: Resolved artifact URI, or ``None`` when using server-side storage.
        is_local: ``True`` when the tracking backend is a local SQLite file.
    """

    tracking_uri: str | None
    resolved_artifact_uri: str | None
    is_local: bool


class MLflowTracker(IExperimentTracker):
    """MLflow implementation of experiment tracker using resource manager pattern.

    Provides MLflow-based experiment tracking with proper resource lifecycle management
    through MLflowResourceManager. Handles client creation, experiment/run management,
    and guaranteed cleanup.

    The tracker should be used as a context manager to ensure proper resource cleanup:

    Example:
        ```python
        from dlkit.runtime.workflows.strategies.tracking import MLflowTracker

        tracker = MLflowTracker()
        tracker.setup_mlflow_config(settings.MLFLOW)

        with tracker:  # Initializes resources
            with tracker.create_run(experiment_name="training") as run:
                run.log_params({"learning_rate": 0.001})
                run.log_metrics({"loss": 0.5}, step=1)
                tracker.log_settings(settings, run)
        # Resources cleaned up automatically
        ```

    Attributes:
        disable_autostart (bool): Skip automatic tracker setup.
    """

    def __init__(self, disable_autostart: bool = False, skip_health_checks: bool = False):
        """Initialize MLflow tracker.

        Args:
            disable_autostart: If True, skip automatic tracker setup.
            skip_health_checks: Kept for compatibility, unused in client-only mode.
        """
        self.disable_autostart = disable_autostart
        self.skip_health_checks = skip_health_checks
        self._resource_manager: MLflowResourceManager | None = None
        self._mlflow_config: Any = None
        self._exit_stack: ExitStack | None = None
        self._tracking_uri: str | None = None
        self._root_dir: Path | None = None

    def __enter__(self) -> "MLflowTracker":
        """Context manager entry - initializes MLflow resources using ExitStack.

        Creates and enters the MLflowResourceManager which handles client
        initialization and experiment setup. Uses ExitStack for nested context
        management to ensure cleanup.

        Returns:
            MLflowTracker: Self for context manager protocol.

        Raises:
            Exception: If resource initialization fails. ExitStack ensures cleanup
                of any partially initialized resources.

        Example:
            ```python
            tracker = MLflowTracker()
            tracker.setup_mlflow_config(settings.MLFLOW)

            with tracker:  # Resources initialized here
                # Use tracker
                pass
            # Resources cleaned up here
            ```
        """
        import os

        logger.debug("MLflow tracker entering (configured={})", self._mlflow_config is not None)

        if self._mlflow_config or os.getenv("MLFLOW_TRACKING_URI"):
            try:
                # Create ExitStack for managing nested context managers
                self._exit_stack = ExitStack()
                self._exit_stack.__enter__()

                # Ensure MLflow resources resolve paths relative to root_dir overrides
                if self._root_dir is not None:
                    ctx = get_current_path_context()
                    has_root_override = bool(ctx and getattr(ctx, "root_dir", None))
                    if not has_root_override:
                        self._exit_stack.enter_context(
                            path_override_context({"root_dir": self._root_dir})
                        )

                # Create and enter resource manager using ExitStack
                logger.debug("Creating resource manager")
                resource_manager = MLflowResourceManager(self._mlflow_config)
                self._resource_manager = self._exit_stack.enter_context(resource_manager)

                logger.debug("MLflow resources initialized")

            except Exception as e:
                logger.error("Failed to initialize MLflow resources: {}", e)
                # Cleanup via ExitStack
                if self._exit_stack:
                    self._exit_stack.__exit__(None, None, None)
                    self._exit_stack = None
                self._resource_manager = None
                self._tracking_uri = None
                raise
        else:
            logger.debug("Skipping resource initialization - no config provided")

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Context manager exit with cleanup via ExitStack.

        Ensures MLflow resources (client and runs) are properly cleaned up.

        Args:
            exc_type: Exception type if an exception occurred, None otherwise.
            exc_val: Exception instance if an exception occurred, None otherwise.
            exc_tb: Exception traceback if an exception occurred, None otherwise.

        Example:
            ```python
            with tracker:
                # Even if exception occurs here
                raise ValueError("Something went wrong")
            # Cleanup still happens
            ```
        """
        logger.debug("MLflowTracker.__exit__ called - exc_type: {}, exc_val: {}", exc_type, exc_val)
        if self._exit_stack:
            try:
                logger.debug("MLflowTracker: Cleaning up MLflow resources via ExitStack")
                self._exit_stack.__exit__(exc_type, exc_val, exc_tb)
                logger.debug("MLflowTracker: ExitStack cleanup completed")
            except Exception as e:
                logger.warning("Failed to clean up MLflow resources: {}", e)
            finally:
                self._exit_stack = None
                self._resource_manager = None
        self._tracking_uri = None
        self._mlflow_config = None
        logger.debug("MLflowTracker.__exit__ completed")

    @contextmanager
    def create_run(
        self,
        experiment_name: str | None = None,
        run_name: str | None = None,
        nested: bool = False,
        tags: dict[str, str] | None = None,
    ) -> AbstractContextManager[IRunContext]:
        """Create MLflow run using resource manager.

        Creates a new MLflow run within the specified experiment. If the experiment
        doesn't exist, it will be created automatically. Supports nested runs for
        hierarchical tracking (e.g., parent optimization run with child trial runs).

        Args:
            experiment_name: Name of experiment to create run under. If None, uses
                default experiment ("DLKit").
            run_name: Optional name for this specific run. If None, MLflow generates
                a unique name automatically.
            nested: If True, creates a child run under the currently active parent run.
                Requires an active parent run context.

        Yields:
            IRunContext: Run context for logging metrics, params, and artifacts.

        Raises:
            RuntimeError: If MLflow not configured (setup_mlflow_config not called).

        Example:
            ```python
            tracker = MLflowTracker()
            tracker.setup_mlflow_config(settings.MLFLOW)

            with tracker:
                # Basic run
                with tracker.create_run(experiment_name="training") as run:
                    run.log_metrics({"loss": 0.5})

                # Nested runs for hyperparameter search
                with tracker.create_run(experiment_name="hp_opt") as parent:
                    for trial in trials:
                        with tracker.create_run(
                            experiment_name="hp_opt", run_name=f"trial_{trial.id}", nested=True
                        ) as child:
                            child.log_params(trial.params)
                            child.log_metrics({"score": trial.score})
            ```
        """
        if not self._resource_manager:
            raise RuntimeError("MLflow not configured - call setup_mlflow_config first")

        exp_name = experiment_name or MLFLOW_DEFAULT_EXPERIMENT

        with self._resource_manager.create_run(
            experiment_name=exp_name,
            run_name=run_name,
            nested=nested,
            tags=tags,
        ) as run_context:
            yield run_context

    def log_settings(self, settings: GeneralSettings, run_context: IRunContext) -> None:
        """Save complete configuration settings as MLflow TOML artifact.

        Creates a temporary TOML file from settings (excluding unset values) and
        uploads it as an artifact to the run. This enables full reproducibility -
        you can recreate the exact training environment from the artifact.

        Args:
            settings: Complete settings object to serialize and log.
            run_context: Active run context to log artifact to.

        Raises:
            RuntimeError: If serialization or artifact logging fails.

        Example:
            ```python
            with tracker.create_run("training") as run:
                # Log full config for reproducibility
                tracker.log_settings(settings, run)
                # Creates "GeneralSettings.toml" artifact in run
            ```
        """
        try:
            from dlkit.tools.io import serialize_config_to_string

            toml_content = serialize_config_to_string(
                settings,
                exclude_unset=True,
                exclude_value_entries=True,
            )
            run_context.log_text(toml_content, "GeneralSettings.toml")
        except Exception as e:
            raise RuntimeError("Couldn't log settings") from e

    def log_model_parameters(
        self, model: Any, run_context: IRunContext, settings: GeneralSettings
    ) -> None:
        """Log model hyperparameters extracted from settings.MODEL.

        Extracts hyperparameters from settings.MODEL (if present) and logs them as
        run parameters. Automatically excludes structural fields (name, module_path,
        checkpoint, shape) to log only actual hyperparameters.

        Args:
            model: Model instance (currently ignored - may be used in future for
                runtime parameter extraction).
            run_context: Active run context to log parameters to.
            settings: Settings object containing MODEL configuration.

        Raises:
            RuntimeError: If parameter extraction or logging fails.

        Example:
            ```python
            # settings.MODEL contains:
            # ModelComponent(name="cnn", hidden_dim=256, num_layers=3, dropout=0.1)

            with tracker.create_run("training") as run:
                tracker.log_model_parameters(model, run, settings)
                # Logs: {"hidden_dim": 256, "num_layers": 3, "dropout": 0.1}
                # Excludes: "name" (structural field)
            ```
        """
        try:
            if settings.MODEL is None:
                return

            # Use settings.MODEL and dump its hyperparameters - so simple!
            params = settings.MODEL.model_dump(exclude_none=True)

            # Remove component-specific fields that aren't hyperparameters
            component_fields = {"name", "module_path", "checkpoint", "shape"}
            hparams = {k: v for k, v in params.items() if k not in component_fields}

            # Only log if we have hyperparameters to log
            if hparams:
                run_context.log_params(hparams)

        except Exception as e:
            raise RuntimeError("Couldn't log settings") from e

    def log_dataset_to_run(
        self, datamodule: Any, run_context: IRunContext, settings: GeneralSettings
    ) -> None:
        """Log dataset lineage to MLflow with structured and artifact fallbacks."""
        dataset = getattr(datamodule, "dataset", None)
        tags = self._build_dataset_tags(settings, dataset)
        sources = self._collect_dataset_sources(settings, dataset)

        structured_logged = self._log_structured_dataset(
            dataset, run_context, settings, tags, sources
        )
        self._log_dataset_manifest_artifact(
            run_context=run_context,
            settings=settings,
            dataset=dataset,
            sources=sources,
            tags=tags,
            structured_logged=structured_logged,
        )

    def _log_structured_dataset(
        self,
        dataset: Any,
        run_context: IRunContext,
        settings: GeneralSettings,
        tags: dict[str, str],
        sources: list[str],
    ) -> bool:
        if dataset is None:
            logger.debug(
                "No dataset found in datamodule, continuing with settings-driven lineage logging"
            )

        dataset_name = self._resolve_dataset_name(settings)
        structured_logger = StructuredDatasetLogger()
        if structured_logger.log(
            dataset=dataset,
            run_context=run_context,
            settings=settings,
            dataset_name=dataset_name,
            sources=sources,
            tags=tags,
        ):
            return True

        logger.warning(
            "Structured MLflow dataset logging unavailable for dataset class '{}' "
            "and current config payload; manifest artifact fallback will be used.",
            type(dataset).__name__ if dataset is not None else "None",
        )
        return False

    def _resolve_dataset_name(self, settings: GeneralSettings) -> str:
        configured_name = getattr(settings.DATASET, "name", None) if settings.DATASET else None
        if configured_name:
            return str(configured_name)
        return "training_data"

    def _build_dataset_tags(self, settings: GeneralSettings, dataset: Any) -> dict[str, str]:
        tags: dict[str, str] = {}
        if settings.DATASET:
            try:
                split_cfg = settings.DATASET.split
                tags["split_test_ratio"] = str(split_cfg.test_ratio)
                tags["split_val_ratio"] = str(split_cfg.val_ratio)
            except Exception:
                pass

            dataset_type = getattr(settings.DATASET, "type", None)
            if dataset_type:
                tags["dataset_type"] = str(dataset_type)

        tags["dataset_class"] = type(dataset).__name__ if dataset is not None else "None"
        return tags

    def _collect_dataset_sources(self, settings: GeneralSettings, dataset: Any) -> list[str]:
        del dataset
        return DatasetSourceCollector().collect(settings)

    def _log_dataset_manifest_artifact(
        self,
        run_context: IRunContext,
        settings: GeneralSettings,
        dataset: Any,
        sources: list[str],
        tags: dict[str, str],
        structured_logged: bool,
    ) -> None:
        try:
            import json
            import hashlib

            fingerprint_payload = json.dumps(sorted(sources), separators=(",", ":"))
            fingerprint = hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()

            manifest = {
                "dataset_name": self._resolve_dataset_name(settings),
                "dataset_class": type(dataset).__name__ if dataset is not None else None,
                "sources": sources,
                "source_count": len(sources),
                "fingerprint": fingerprint,
                "tags": tags,
                "structured_mlflow_dataset_logged": structured_logged,
            }

            run_context.log_text(
                json.dumps(manifest, indent=2, sort_keys=True),
                "lineage/dataset_manifest.json",
            )
            run_context.set_tag("dataset_manifest_artifact", "lineage")
            run_context.set_tag("dataset_source_count", str(len(sources)))
            run_context.set_tag("dataset_fingerprint", fingerprint)
        except Exception as e:
            logger.warning("Failed to log dataset manifest artifact: {}", e)

    def setup_mlflow_config(
        self,
        mlflow_config: Any,
        *,
        root_dir: Path | str | None = None,
    ) -> TrackingSetupResult:
        """Configure MLflow tracking - stores config for deferred resource initialization.

        Stores the MLflow configuration for later use. Actual resource initialization
        (client creation and URI setup) is deferred until __enter__() to follow
        proper context manager protocol and ensure cleanup.

        Args:
            mlflow_config: MLflow configuration settings (typically from settings.MLFLOW).
                Must have 'enabled' attribute. If enabled=False, tracking is skipped.

        Returns:
            tuple[str | None, dict | None]: Tuple of (tracking_uri, None).

        Example:
            ```python
            from dlkit.tools.config import GeneralSettings

            settings = GeneralSettings.from_toml("config.toml")
            tracker = MLflowTracker()

            # Just stores config - no resource initialization yet
            result = tracker.setup_mlflow_config(settings.MLFLOW)
            result.tracking_uri  # May be None

            # Resources initialized here
            with tracker:
                with tracker.create_run("exp") as run:
                    run.log_metrics({"loss": 0.5})
            ```
        """
        import os

        self._root_dir = self._determine_root_dir(root_dir)
        self._mlflow_config = mlflow_config
        self._tracking_uri = None

        env_uri = os.getenv("MLFLOW_TRACKING_URI")

        # Skip setup if disabled or neither config nor env var is present
        if self.disable_autostart or (not mlflow_config and not env_uri):
            logger.debug("MLflow not configured or autostart disabled")
            return TrackingSetupResult(tracking_uri=None, resolved_artifact_uri=None, is_local=False)

        from .uri_resolver import resolve_mlflow_uris

        resolved = resolve_mlflow_uris()
        self._tracking_uri = resolved.tracking_uri
        logger.debug("MLflow config stored - will initialize in context entry")
        return TrackingSetupResult(
            tracking_uri=resolved.tracking_uri,
            resolved_artifact_uri=resolved.artifact_uri,
            is_local=resolved.scheme == "sqlite",
        )

    def _determine_root_dir(self, candidate: Path | str | None) -> Path | None:
        if candidate is not None:
            return Path(candidate).resolve()

        ctx = get_current_path_context()
        root_dir_val = getattr(ctx, "root_dir", None) if ctx else None
        if root_dir_val is not None:
            return Path(str(root_dir_val)).resolve()

        return None
