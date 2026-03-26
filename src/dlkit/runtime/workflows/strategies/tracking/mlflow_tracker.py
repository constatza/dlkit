"""MLflow adapter implementing tracking abstractions."""

from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack
from pathlib import Path
from typing import Any

from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.workflow_configs import OptimizationWorkflowConfig, TrainingWorkflowConfig
from dlkit.tools.io.path_context import (
    get_current_path_context,
    path_override_context,
)
from dlkit.tools.utils.logging_config import get_logger

from .backend import LocalSqliteBackend, TrackingBackend, select_backend
from .dataset_lineage import DatasetSourceCollector, StructuredDatasetLogger
from .interfaces import IExperimentTracker, IRunContext
from .mlflow_resource_manager import MLflowResourceManager

type _WorkflowSettings = GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig

logger = get_logger(__name__)

# MLflow integration constants
MLFLOW_DEFAULT_EXPERIMENT = "DLKit"


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
        tracker.configure(settings.MLFLOW)

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

    def __init__(
        self,
        disable_autostart: bool = False,
        probe: Callable[[], bool] | None = None,
    ):
        """Initialize MLflow tracker.

        Args:
            disable_autostart: If True, skip automatic tracker setup.
            probe: Optional callable to detect a local MLflow server.
                Defaults to ``local_host_alive`` from ``uri_resolver``.
        """
        self.disable_autostart = disable_autostart
        self._probe = probe
        self._resource_manager: MLflowResourceManager | None = None
        self._mlflow_config: Any = None
        self._exit_stack: ExitStack | None = None
        self._backend: TrackingBackend | None = None
        self._root_dir: Path | None = None

    def __enter__(self) -> MLflowTracker:
        """Context manager entry - initializes MLflow resources using ExitStack.

        Creates and enters the MLflowResourceManager which handles client
        initialization and experiment setup. Uses ExitStack for nested context
        management to ensure cleanup.

        Returns:
            MLflowTracker: Self for context manager protocol.

        Raises:
            Exception: If resource initialization fails.
        """
        logger.debug("MLflow tracker entering (configured={})", self._mlflow_config is not None)

        if self._mlflow_config and not self.disable_autostart:
            try:
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

                logger.debug("Selecting tracking backend")
                self._backend = select_backend(probe=self._probe)

                logger.debug("Creating resource manager")
                resource_manager = MLflowResourceManager(self._mlflow_config, self._backend)
                self._resource_manager = self._exit_stack.enter_context(resource_manager)

                logger.debug("MLflow resources initialized")

            except Exception as e:
                logger.error("Failed to initialize MLflow resources: {}", e)
                if self._exit_stack:
                    self._exit_stack.__exit__(None, None, None)
                    self._exit_stack = None
                self._resource_manager = None
                self._backend = None
                raise
        else:
            logger.debug(
                "Skipping resource initialization - no config provided or autostart disabled"
            )

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Context manager exit with cleanup via ExitStack."""
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
                self._backend = None
        self._mlflow_config = None
        logger.debug("MLflowTracker.__exit__ completed")

    def create_run(
        self,
        experiment_name: str | None = None,
        run_name: str | None = None,
        nested: bool = False,
        tags: dict[str, str] | None = None,
    ) -> AbstractContextManager[IRunContext]:
        """Create MLflow run using resource manager.

        Args:
            experiment_name: Name of experiment to create run under.
            run_name: Optional name for this specific run.
            nested: If True, creates a child run under the currently active parent run.
            tags: Optional tags to attach to the run.

        Returns:
            AbstractContextManager[IRunContext]: Context manager yielding active run context.

        Raises:
            RuntimeError: If MLflow not configured (configure not called).
        """
        if not self._resource_manager:
            raise RuntimeError("MLflow not configured - call configure() before entering context")

        exp_name = experiment_name or MLFLOW_DEFAULT_EXPERIMENT

        return self._resource_manager.create_run(
            experiment_name=exp_name,
            run_name=run_name,
            nested=nested,
            tags=tags,
        )

    def get_tracking_uri(self) -> str | None:
        """Return the resolved tracking URI, or None if not initialized.

        Returns:
            Tracking URI string or None.
        """
        return self._backend.tracking_uri() if self._backend is not None else None

    def is_local(self) -> bool:
        """Return True when using a local SQLite backend.

        Returns:
            True if the backend is ``LocalSqliteBackend``.
        """
        return isinstance(self._backend, LocalSqliteBackend)

    def configure(
        self,
        mlflow_config: Any,
        *,
        root_dir: Path | str | None = None,
    ) -> None:
        """Store MLflow config — no side effects. Resolution deferred to __enter__.

        Args:
            mlflow_config: MLflow configuration settings.
            root_dir: Optional root directory override for path resolution.
        """
        self._mlflow_config = mlflow_config
        self._root_dir = self._determine_root_dir(root_dir)
        logger.debug("MLflow config stored - will initialize in context entry")

    def log_settings(self, settings: _WorkflowSettings, run_context: IRunContext) -> None:
        """Save complete configuration settings as MLflow TOML artifact.

        Args:
            settings: Complete settings object to serialize and log.
            run_context: Active run context to log artifact to.

        Raises:
            RuntimeError: If serialization or artifact logging fails.
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
        self, model: Any, run_context: IRunContext, settings: _WorkflowSettings
    ) -> None:
        """Log model hyperparameters extracted from settings.MODEL.

        Args:
            model: Model instance (currently ignored).
            run_context: Active run context to log parameters to.
            settings: Settings object containing MODEL configuration.

        Raises:
            RuntimeError: If parameter extraction or logging fails.
        """
        try:
            if settings.MODEL is None:
                return

            params = settings.MODEL.model_dump(exclude_none=True)

            component_fields = {"name", "module_path", "checkpoint", "shape"}
            hparams = {k: v for k, v in params.items() if k not in component_fields}

            if hparams:
                run_context.log_params(hparams)

        except Exception as e:
            raise RuntimeError("Couldn't log settings") from e

    def log_dataset_to_run(
        self, datamodule: Any, run_context: IRunContext, settings: _WorkflowSettings
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
        settings: _WorkflowSettings,
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

    def _resolve_dataset_name(self, settings: _WorkflowSettings) -> str:
        configured_name = getattr(settings.DATASET, "name", None) if settings.DATASET else None
        if configured_name:
            return str(configured_name)
        return "training_data"

    def _build_dataset_tags(self, settings: _WorkflowSettings, dataset: Any) -> dict[str, str]:
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

    def _collect_dataset_sources(self, settings: _WorkflowSettings, dataset: Any) -> list[str]:
        del dataset
        return DatasetSourceCollector().collect(settings)

    def _log_dataset_manifest_artifact(
        self,
        run_context: IRunContext,
        settings: _WorkflowSettings,
        dataset: Any,
        sources: list[str],
        tags: dict[str, str],
        structured_logged: bool,
    ) -> None:
        try:
            import hashlib
            import json

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

    def _determine_root_dir(self, candidate: Path | str | None) -> Path | None:
        if candidate is not None:
            return Path(candidate).resolve()

        ctx = get_current_path_context()
        root_dir_val = getattr(ctx, "root_dir", None) if ctx else None
        if root_dir_val is not None:
            return Path(str(root_dir_val)).resolve()

        return None
