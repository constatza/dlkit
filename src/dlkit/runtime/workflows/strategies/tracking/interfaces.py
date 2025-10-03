"""Tracking abstractions following DIP."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from dlkit.tools.config import GeneralSettings


class IRunContext(ABC):
    """Context for an active tracking run.

    Provides methods for logging metrics, parameters, artifacts, and tags to an
    active experiment tracking run. Implementations handle the actual persistence
    to tracking backends (e.g., MLflow).

    Example:
        ```python
        with tracker.create_run("my_experiment") as run_context:
            # Log hyperparameters
            run_context.log_params({"learning_rate": 0.001, "batch_size": 32})

            # Log training metrics
            for epoch in range(10):
                loss = train_epoch()
                run_context.log_metrics({"loss": loss}, step=epoch)

            # Log model artifact
            run_context.log_artifact(Path("model.pth"), artifact_dir="models")

            # Tag the run
            run_context.set_tag("status", "completed")
        ```
    """

    @property
    @abstractmethod
    def run_id(self) -> str:
        """Get the unique identifier for this tracking run.

        Returns:
            str: Unique run identifier (e.g., MLflow run UUID).

        Example:
            ```python
            with tracker.create_run("exp") as run:
                print(f"Run ID: {run.run_id}")
            ```
        """
        raise NotImplementedError

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to the active run.

        Args:
            metrics: Dictionary mapping metric names to numeric values.
            step: Optional step number for time-series metrics (e.g., epoch, iteration).

        Example:
            ```python
            # Log single metric
            run_context.log_metrics({"accuracy": 0.95})

            # Log multiple metrics with step
            run_context.log_metrics(
                {"loss": 0.3, "val_loss": 0.35, "lr": 0.001},
                step=42
            )
            ```
        """
        raise NotImplementedError

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters (hyperparameters) to the active run.

        Parameters are typically configuration values that don't change during training.

        Args:
            params: Dictionary mapping parameter names to values (str, int, float, bool).

        Example:
            ```python
            run_context.log_params({
                "learning_rate": 0.001,
                "batch_size": 32,
                "optimizer": "adam",
                "num_layers": 3,
                "dropout": 0.1
            })
            ```
        """
        raise NotImplementedError

    @abstractmethod
    def log_artifact(self, artifact_path: Path, artifact_dir: str = "") -> None:
        """Log an artifact file to the active run.

        Artifacts are files produced by the run (models, plots, datasets, etc.).

        Args:
            artifact_path: Path to the local file to upload.
            artifact_dir: Optional subdirectory within run's artifact store.

        Example:
            ```python
            # Log model checkpoint
            run_context.log_artifact(Path("best_model.pth"), artifact_dir="models")

            # Log training plot
            run_context.log_artifact(Path("loss_curve.png"), artifact_dir="plots")

            # Log to root artifact directory
            run_context.log_artifact(Path("config.yaml"))
            ```
        """
        raise NotImplementedError

    @abstractmethod
    def set_tag(self, key: str, value: str) -> None:
        """Set a tag (key-value metadata) on the active run.

        Tags are string key-value pairs for organizing and filtering runs.

        Args:
            key: Tag name/key.
            value: Tag value (will be converted to string).

        Example:
            ```python
            run_context.set_tag("status", "completed")
            run_context.set_tag("model_type", "cnn")
            run_context.set_tag("dataset_version", "v2.1")
            run_context.set_tag("git_commit", "a3f2d1b")
            ```
        """
        raise NotImplementedError


class IExperimentTracker(ABC):
    """Abstract experiment tracker following Dependency Inversion Principle.

    Defines the interface for experiment tracking systems. Implementations provide
    concrete tracking backends (MLflow, Weights & Biases, etc.). This abstraction
    allows workflow code to depend on the interface rather than specific implementations.

    Implementations should be used as context managers to ensure proper resource cleanup.

    Example:
        ```python
        # Code works with any IExperimentTracker implementation
        def train_with_tracking(tracker: IExperimentTracker, settings: GeneralSettings):
            with tracker:
                with tracker.create_run(experiment_name="training") as run:
                    tracker.log_settings(settings, run)
                    run.log_metrics({"accuracy": 0.95})

        # Use MLflow implementation
        train_with_tracking(MLflowTracker(), settings)

        # Or null implementation (no tracking)
        train_with_tracking(NullTracker(), settings)
        ```
    """

    @abstractmethod
    def create_run(
        self,
        experiment_name: str | None = None,
        run_name: str | None = None,
        nested: bool = False,
    ) -> AbstractContextManager[IRunContext]:
        """Create a tracking run context.

        Args:
            experiment_name: Name of experiment to organize runs under. If None, uses default.
            run_name: Optional name for this specific run. If None, auto-generated.
            nested: Whether this is a child run under an active parent run.

        Returns:
            AbstractContextManager[IRunContext]: Context manager yielding active run context.

        Example:
            ```python
            # Basic run
            with tracker.create_run(experiment_name="training") as run:
                run.log_metrics({"loss": 0.5})

            # Nested runs for hyperparameter search
            with tracker.create_run(experiment_name="hp_search") as parent:
                for trial in trials:
                    with tracker.create_run(
                        experiment_name="hp_search",
                        run_name=f"trial_{trial.id}",
                        nested=True
                    ) as child:
                        child.log_params(trial.params)
            ```
        """
        raise NotImplementedError

    @abstractmethod
    def log_settings(self, settings: GeneralSettings, run_context: IRunContext) -> None:
        """Log complete configuration settings to the run.

        Typically saves settings as an artifact (e.g., TOML file) for reproducibility.

        Args:
            settings: Complete settings object to log.
            run_context: Active run context to log to.

        Raises:
            RuntimeError: If logging fails.

        Example:
            ```python
            with tracker.create_run("experiment") as run:
                # Log full config for reproducibility
                tracker.log_settings(settings, run)
            ```
        """
        raise NotImplementedError

    @abstractmethod
    def log_model_parameters(
        self, model: Any, run_context: IRunContext, settings: GeneralSettings
    ) -> None:
        """Log model hyperparameters extracted from settings.

        Extracts hyperparameters from settings.MODEL and logs them as run parameters.
        Excludes structural fields (name, module_path, checkpoint, shape).

        Args:
            model: Model instance (currently ignored, may be used in future).
            run_context: Active run context to log to.
            settings: Settings containing MODEL configuration with hyperparameters.

        Raises:
            RuntimeError: If parameter extraction or logging fails.

        Example:
            ```python
            # Settings contains model hyperparameters
            settings.MODEL = ModelComponent(
                name="my_model",
                hidden_dim=256,
                num_layers=3,
                dropout=0.1
            )

            with tracker.create_run("training") as run:
                # Logs: {"hidden_dim": 256, "num_layers": 3, "dropout": 0.1}
                tracker.log_model_parameters(model, run, settings)
            ```
        """
        raise NotImplementedError


class NullRunContext(IRunContext):
    """Null object implementation of IRunContext for when tracking is disabled.

    Provides safe no-op implementations of all IRunContext methods. Allows code to
    call tracking methods without checking if tracking is enabled, following the
    Null Object Pattern.

    Example:
        ```python
        # Code doesn't need to check if tracking is enabled
        def train(run_context: IRunContext):
            run_context.log_params({"lr": 0.001})  # Works with both real and null context
            run_context.log_metrics({"loss": 0.5})

        # Works with null context
        train(NullRunContext())
        ```
    """

    @property
    def run_id(self) -> str:
        """Return a placeholder run ID.

        Returns:
            str: Always returns "null-run-id" as a placeholder.
        """
        return "null-run-id"

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """No-op metrics logging.

        Args:
            metrics: Ignored.
            step: Ignored.
        """
        pass

    def log_params(self, params: dict[str, Any]) -> None:
        """No-op parameter logging.

        Args:
            params: Ignored.
        """
        pass

    def log_artifact(self, artifact_path: Path, artifact_dir: str = "") -> None:
        """No-op artifact logging.

        Args:
            artifact_path: Ignored.
            artifact_dir: Ignored.
        """
        pass

    def set_tag(self, key: str, value: str) -> None:
        """No-op tag setting.

        Args:
            key: Ignored.
            value: Ignored.
        """
        pass


class NullTracker(IExperimentTracker):
    """Null object implementation of IExperimentTracker for when tracking is disabled.

    Provides safe no-op implementations following the Null Object Pattern. Eliminates
    the need for conditional logic throughout the codebase - the same code works
    whether tracking is enabled or disabled.

    This is the recommended approach for disabling tracking rather than using if/else
    statements or None checks.

    Example:
        ```python
        # Select tracker based on configuration
        if settings.MLFLOW.enabled:
            tracker: IExperimentTracker = MLflowTracker()
        else:
            tracker: IExperimentTracker = NullTracker()

        # Same code works with both - no conditionals needed!
        with tracker:
            with tracker.create_run("experiment") as run:
                tracker.log_settings(settings, run)
                run.log_metrics({"accuracy": 0.95})
        ```
    """

    def __enter__(self) -> "NullTracker":
        """Context manager entry for null tracker.

        Returns:
            NullTracker: Self for context manager protocol.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Context manager exit for null tracker.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """
        pass

    def create_run(
        self,
        experiment_name: str | None = None,
        run_name: str | None = None,
        nested: bool = False,
    ) -> AbstractContextManager[IRunContext]:
        """Create a null context manager that provides a NullRunContext.

        Args:
            experiment_name: Ignored.
            run_name: Ignored.
            nested: Ignored.

        Returns:
            AbstractContextManager[IRunContext]: Context manager yielding NullRunContext.

        Example:
            ```python
            tracker = NullTracker()
            with tracker.create_run("exp") as run:
                run.log_metrics({"loss": 0.5})  # No-op, but safe
            ```
        """
        from contextlib import contextmanager

        @contextmanager
        def _null_context():
            yield NullRunContext()

        return _null_context()

    def log_settings(self, settings: GeneralSettings, run_context: IRunContext) -> None:
        """No-op settings logging.

        Args:
            settings: Ignored.
            run_context: Ignored.
        """
        pass

    def log_model_parameters(
        self, model: Any, run_context: IRunContext, settings: GeneralSettings
    ) -> None:
        """No-op model parameter logging.

        Args:
            model: Ignored.
            run_context: Ignored.
            settings: Ignored.
        """
        pass
