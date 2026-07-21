"""Core API functions for training, optimization, and inference."""

from __future__ import annotations

from pathlib import Path

from lightning.pytorch import LightningDataModule

from dlkit.common import (
    OptimizationResult,
    TrainingResult,
)
from dlkit.common.results import ConvergenceResult
from dlkit.engine.workflows.entrypoints._settings import WorkflowSettings
from dlkit.engine.workflows.factories.inference_data_factory import (
    build_inference_datamodule as _build_inference_datamodule,
)
from dlkit.infrastructure.config.job_config import InferenceJobConfig
from dlkit.interfaces.api.adapters import EngineWorkflowExecutor
from dlkit.interfaces.api.domain.override_types import (
    ConvergenceOverrides,
    OptimizationOverrides,
    TrainingOverrides,
)

_executor: EngineWorkflowExecutor = EngineWorkflowExecutor()


def train(
    settings: WorkflowSettings,
    overrides: TrainingOverrides | None = None,
    *,
    mlflow: bool = False,
) -> TrainingResult:
    """Run training with optional overrides.

    Args:
        settings: Training workflow configuration settings.
        overrides: Optional training overrides (paths coerced to Path objects).
        mlflow: If True, force MLflow tracking behavior for this execution.

    Returns:
        TrainingResult containing trained model state and metrics.
    """
    return _executor.train(
        settings,
        overrides=overrides,
        mlflow=mlflow,
    )


def build_inference_datamodule(
    settings: InferenceJobConfig,
    *,
    checkpoint_override: Path | str | None = None,
) -> LightningDataModule:
    """Build a datamodule for inference batch iteration.

    No training wrapper, no loss, no optimizer. Only run/experiment, data sections.
    Pure function: no class, no side effects beyond datamodule construction.

    Args:
        settings: Inference job configuration with data section.
        checkpoint_override: Checkpoint path supplied directly by the caller,
            used to auto-locate a colocated split file when
            ``data.splits.filepath`` is unset. Takes precedence over
            ``settings.model.checkpoint``.

    Returns:
        Configured LightningDataModule ready for predict_dataloader iteration.

    Raises:
        ValueError: If data section is not configured.
    """
    return _build_inference_datamodule(settings, checkpoint_override=checkpoint_override)


# REMOVED: Old infer() and predict_with_config() functions
# Use the new load_model() API instead:
#
#   from dlkit import load_model
#   predictor = load_model("model.ckpt", device="cuda")
#   result = predictor.predict(inputs)
#
# See documentation for migration guide.


def converge(
    settings: WorkflowSettings,
    overrides: ConvergenceOverrides | None = None,
    *,
    mlflow: bool = False,
) -> ConvergenceResult:
    """Run a sample-size convergence study with optional overrides.

    Args:
        settings: Convergence workflow configuration settings.
        overrides: Optional convergence overrides (sizes, repeats, target).
        mlflow: If True, force MLflow tracking behavior for this execution.

    Returns:
        ConvergenceResult with convergence points, n_star, and tracking metadata.
    """
    return _executor.converge(
        settings,
        overrides=overrides,
        mlflow=mlflow,
    )


def optimize(
    settings: WorkflowSettings,
    overrides: OptimizationOverrides | None = None,
    *,
    mlflow: bool = False,
) -> OptimizationResult:
    """Run Optuna hyperparameter optimization with optional overrides.

    Args:
        settings: Optimization workflow configuration settings.
        overrides: Optional optimization overrides (paths coerced to Path objects).
        mlflow: If True, force MLflow tracking behavior for this execution.

    Returns:
        OptimizationResult containing best model and trial history.
    """
    return _executor.optimize(
        settings,
        overrides=overrides,
        mlflow=mlflow,
    )
