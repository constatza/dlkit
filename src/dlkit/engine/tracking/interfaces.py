"""Tracking abstractions following DIP."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

from dlkit.common.hooks import ParamValue
from dlkit.engine.artifacts import IMetricSink as IMetricSink  # noqa: PLC0414
from dlkit.infrastructure.config.tracking_settings import TrackingSettings

if TYPE_CHECKING:
    from mlflow.data.dataset import Dataset as MlflowDataset
    from mlflow.entities import Dataset as MlflowDatasetEntity
    from mlflow.models import ModelSignature
    from sklearn.base import BaseEstimator
    from torch import nn

# Extensible tracking payload sum types.
# These are sum types, not aliases: they enumerate currently supported
# MLflow-facing payload shapes and are intended to grow for other backends.
type LoggedDataset = MlflowDataset | MlflowDatasetEntity
type LoggedModel = nn.Module | BaseEstimator


class IRunContext(IMetricSink, ABC):
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
                run.run_id is not None
            ```
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def experiment_id(self) -> str | None:
        """Get the experiment identifier for this tracking run."""
        raise NotImplementedError

    @property
    @abstractmethod
    def tracking_uri(self) -> str | None:
        """Get the resolved tracking URI for this tracking run."""
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
            run_context.log_metrics({"loss": 0.3, "val_loss": 0.35, "lr": 0.001}, step=42)
            ```
        """
        raise NotImplementedError

    @abstractmethod
    def log_params(self, params: Mapping[str, ParamValue]) -> None:
        """Log parameters (hyperparameters) to the active run.

        Parameters are typically configuration values that don't change during training.

        Args:
            params: Dictionary mapping parameter names to values (str, int, float, bool).

        Example:
            ```python
            run_context.log_params(
                {
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "optimizer": "adam",
                    "num_layers": 3,
                    "dropout": 0.1,
                }
            )
            ```
        """
        raise NotImplementedError

    @abstractmethod
    def log_artifact_content(self, content: str | bytes, artifact_file: str) -> None:
        """Log small artifact content without writing an intermediate file.

        Args:
            content: Artifact content to log.
            artifact_file: Filename (with optional subdirectory) within the run's
                artifact store, e.g. ``"config/settings.toml"`` or ``"manifest.json"``.

        Example:
            ```python
            run_context.log_artifact_content('{"key": "value"}', "lineage/manifest.json")
            run_context.log_artifact_content(
                '[tracking]\\nbackend = "mlflow"', "config/settings.toml"
            )
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

    @abstractmethod
    def log_dataset(
        self,
        dataset: LoggedDataset,
        context: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Log a dataset to the active run.

        Logs dataset metadata for reproducibility and lineage tracking. Accepts MLflow
        dataset objects created via mlflow.data.from_numpy(), mlflow.data.from_pandas(), etc.

        Args:
            dataset: MLflow dataset object (e.g., from mlflow.data.from_numpy()).
            context: Optional context string describing dataset usage (e.g., "training", "validation").
            tags: Optional dictionary of tags to associate with the dataset.

        Example:
            ```python
            import mlflow.data
            import numpy as np

            # Create MLflow dataset
            features = np.random.rand(100, 10)
            targets = np.random.rand(100, 1)
            dataset = mlflow.data.from_numpy(
                features=features, targets=targets, name="training_data", source="data/train.npy"
            )

            # Log to run
            run_context.log_dataset(dataset, context="training")
            ```
        """
        raise NotImplementedError

    @abstractmethod
    def log_model(
        self,
        model: LoggedModel,
        artifact_path: str,
        *,
        registered_model_name: str | None = None,
        signature: ModelSignature | None = None,
        # Intentionally opaque foreign backend payload forwarded to MLflow.
        input_example: object | None = None,
    ) -> str | None:
        """Log a model artifact to the active run.

        Args:
            model: Model object to log.
            artifact_path: Artifact subpath under the run.
            registered_model_name: Optional registered model target name.
            signature: Optional model signature.
            input_example: Optional model input example.

        Returns:
            Model URI for the logged artifact if available, otherwise None.
        """
        raise NotImplementedError

    @abstractmethod
    def get_latest_model_version(
        self,
        model_name: str,
        *,
        run_id: str | None = None,
        artifact_path: str | None = None,
    ) -> int | None:
        """Get the latest numeric version for a registered model.

        Args:
            model_name: Registered model name.
            run_id: Optional run identifier to constrain the version lookup.
            artifact_path: Optional artifact path hint for additional filtering.

        Returns:
            Latest version number, or None when unavailable.
        """
        raise NotImplementedError

    @abstractmethod
    def set_model_alias(self, model_name: str, alias: str, version: int) -> None:
        """Set a registered model alias.

        Args:
            model_name: Registered model name.
            alias: Alias value (e.g. "latest", "candidate").
            version: Target version.
        """
        raise NotImplementedError

    @abstractmethod
    def set_model_version_tag(
        self,
        model_name: str,
        version: int,
        key: str,
        value: str,
    ) -> None:
        """Set a tag on a registered model version."""
        raise NotImplementedError

    def is_active(self) -> bool:
        """Return True when this is a real (non-null) tracking run.

        Returns:
            True for live tracking runs; False for null/stub contexts.
        """
        return True


class IExperimentTracker(ABC):
    """Abstract experiment tracker following Dependency Inversion Principle.

    Defines the interface for experiment tracking systems. Implementations provide
    concrete tracking backends (MLflow, Weights & Biases, etc.). This abstraction
    allows workflow code to depend on the interface rather than specific implementations.

    Implementations should be used as context managers to ensure proper resource cleanup.

    Example:
        ```python
        # Code works with any IExperimentTracker implementation
        def train_with_tracking(tracker: IExperimentTracker, settings: JobConfig):
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
    def __enter__(self) -> IExperimentTracker:
        """Enter the tracker context manager, initialising resources."""
        raise NotImplementedError

    @abstractmethod
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        """Exit the tracker context manager, releasing resources."""
        raise NotImplementedError

    @abstractmethod
    def create_run(
        self,
        experiment_name: str | None = None,
        run_name: str | None = None,
        nested: bool = False,
        tags: dict[str, str] | None = None,
    ) -> AbstractContextManager[IRunContext]:
        """Create a tracking run context.

        Args:
            experiment_name: Name of experiment to organize runs under. If None, uses default.
            run_name: Optional name for this specific run. If None, auto-generated.
            nested: Whether this is a child run under an active parent run.
            tags: Optional key-value tags to attach to the run.

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
                        experiment_name="hp_search", run_name=f"trial_{trial.id}", nested=True
                    ) as child:
                        child.log_params(trial.params)
            ```
        """
        raise NotImplementedError

    @abstractmethod
    def get_tracking_uri(self) -> str | None:
        """Return the resolved tracking URI, or None if not initialized.

        Returns:
            Tracking URI string or None.
        """
        raise NotImplementedError

    @abstractmethod
    def is_local(self) -> bool:
        """Return True when the backend stores data on the local filesystem.

        Returns:
            True if the backend is local (e.g., SQLite / local directory).
        """
        raise NotImplementedError

    def configure(self, config: TrackingSettings) -> None:
        """Configure the tracker with backend-specific settings (no-op by default).

        Concrete backends override this to apply configuration before any runs are created.

        Args:
            config: Backend-specific configuration object.
        """
        return None


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
        """Return empty string — null context has no real run ID."""
        return ""

    @property
    def experiment_id(self) -> str | None:
        """Return None (null tracker has no experiment)."""
        return None

    @property
    def tracking_uri(self) -> str | None:
        """Return None (null tracker has no backend)."""
        return None

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """No-op metrics logging.

        Args:
            metrics: Ignored.
            step: Ignored.
        """

    def log_params(self, params: Mapping[str, ParamValue]) -> None:
        """No-op parameter logging.

        Args:
            params: Ignored.
        """

    def log_artifact_content(self, content: str | bytes, artifact_file: str) -> None:
        """No-op artifact-content logging.

        Args:
            content: Ignored.
            artifact_file: Ignored.
        """

    def log_artifact(self, artifact_path: Path, artifact_dir: str = "") -> None:
        """No-op artifact logging.

        Args:
            artifact_path: Ignored.
            artifact_dir: Ignored.
        """

    def set_tag(self, key: str, value: str) -> None:
        """No-op tag setting.

        Args:
            key: Ignored.
            value: Ignored.
        """

    def log_dataset(
        self,
        dataset: LoggedDataset,
        context: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        """No-op dataset logging.

        Args:
            dataset: Ignored.
            context: Ignored.
            tags: Ignored.
        """

    def log_model(
        self,
        model: LoggedModel,
        artifact_path: str,
        *,
        registered_model_name: str | None = None,
        signature: ModelSignature | None = None,
        # Intentionally opaque foreign backend payload forwarded to MLflow.
        input_example: object | None = None,
    ) -> str | None:
        """No-op model logging."""
        return None

    def get_latest_model_version(
        self,
        model_name: str,
        *,
        run_id: str | None = None,
        artifact_path: str | None = None,
    ) -> int | None:
        """No-op latest model version lookup."""
        return None

    def set_model_alias(self, model_name: str, alias: str, version: int) -> None:
        """No-op model alias update."""

    def set_model_version_tag(
        self,
        model_name: str,
        version: int,
        key: str,
        value: str,
    ) -> None:
        """No-op model version tag update."""

    def is_active(self) -> bool:
        """Return False — null context is not a real tracking run."""
        return False


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
        if settings.tracking.backend == "mlflow":
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

    def __enter__(self) -> NullTracker:
        """Context manager entry for null tracker.

        Returns:
            NullTracker: Self for context manager protocol.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit for null tracker.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """

    def create_run(
        self,
        experiment_name: str | None = None,
        run_name: str | None = None,
        nested: bool = False,
        tags: dict[str, str] | None = None,
    ) -> AbstractContextManager[IRunContext]:
        """Create a null context manager that provides a NullRunContext.

        Args:
            experiment_name: Ignored.
            run_name: Ignored.
            nested: Ignored.
            tags: Ignored.

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

    def get_tracking_uri(self) -> str | None:
        """Return None — null tracker has no backend."""
        return None

    def is_local(self) -> bool:
        """Return False — null tracker has no backend."""
        return False
