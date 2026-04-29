"""Core API functions for training, optimization, and inference."""

from __future__ import annotations

from lightning.pytorch import LightningDataModule

from dlkit.common import (
    OptimizationResult,
    TrainingResult,
)
from dlkit.engine.workflows.entrypoints._settings import WorkflowSettings
from dlkit.engine.workflows.factories.inference_data_factory import (
    build_inference_datamodule as _build_inference_datamodule,
)
from dlkit.infrastructure.config.workflow_configs import InferenceWorkflowConfig
from dlkit.interfaces.api.adapters import EngineWorkflowExecutor
from dlkit.interfaces.api.domain.override_types import (
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
        mlflow: If True, ensure MLFLOW section exists in settings.

    Returns:
        TrainingResult containing trained model state and metrics.
    """
    return _executor.train(
        settings,
        overrides=overrides,
        mlflow=mlflow,
    )


def build_inference_datamodule(settings: InferenceWorkflowConfig) -> LightningDataModule:
    """Build a datamodule for inference batch iteration.

    No training wrapper, no loss, no optimizer. Only SESSION, DATASET, DATAMODULE.
    Pure function: no class, no side effects beyond datamodule construction.

    Args:
        settings: Inference workflow configuration with DATASET and DATAMODULE sections.

    Returns:
        Configured LightningDataModule ready for predict_dataloader iteration.

    Raises:
        ValueError: If DATASET or DATAMODULE sections are not configured.
    """
    return _build_inference_datamodule(settings)


# REMOVED: Old infer() and predict_with_config() functions
# Use the new load_model() API instead:
#
#   from dlkit import load_model
#   predictor = load_model("model.ckpt", device="cuda")
#   result = predictor.predict(inputs)
#
# See documentation for migration guide.


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
        mlflow: If True, ensure MLFLOW section exists in settings.

    Returns:
        OptimizationResult containing best model and trial history.
    """
    return _executor.optimize(
        settings,
        overrides=overrides,
        mlflow=mlflow,
    )
