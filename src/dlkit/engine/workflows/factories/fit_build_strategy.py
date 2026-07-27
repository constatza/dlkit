"""Build strategy for one-shot, non-gradient ``Fittable`` models."""

from __future__ import annotations

from typing import cast

from lightning.pytorch import LightningModule

from dlkit.engine.adapters.lightning.base import _build_model_from_settings
from dlkit.engine.artifacts import RuntimeArtifactManifest
from dlkit.engine.training.components import RuntimeComponents
from dlkit.infrastructure.config.job_config import FitJobConfig

from .build_strategy import IBuildStrategy, WorkflowSettings, _build_datamodule
from .dataset_builder import DatasetBuilder
from .module_defaults import with_runtime_module_defaults


class FitBuildStrategy(IBuildStrategy):
    """Build strategy for models fit via a single deterministic, non-gradient call.

    Builds the dataset/datamodule the same way the flexible-array path does
    (raw ``data.features``/``data.targets`` entries, no loss/metric-driven
    feature-dependency selection — a ``FitJobConfig`` has no ``training``
    section to select against), but skips the Lightning wrapper entirely:
    ``WrapperFactory.create_*_wrapper`` requires an ``IOptimizationController``,
    which a closed-form-fit model has no use for. The raw model is
    constructed directly and returned as-is; it must already be a
    ``LightningModule`` implementing the ``Fittable`` protocol (a
    model-side contract, not something this strategy enforces). Its actual
    ``fit()`` call is deferred to ``OneShotFitExecutor`` since
    ``RuntimeComponents.trainer`` is ``None`` here — there is no ``Trainer``
    to drive it.
    """

    def __init__(self, dataset_builder: DatasetBuilder | None = None) -> None:
        self._dataset_builder = dataset_builder or DatasetBuilder()

    def can_handle(self, settings: WorkflowSettings) -> bool:
        return isinstance(settings, FitJobConfig)

    def _build_core(self, settings: WorkflowSettings) -> RuntimeComponents:
        from dlkit.engine.workflows.factories._dataset_helpers import flexible_dataset_overrides

        data = settings.data
        if data is None:
            raise ValueError("DATASET settings are required but not configured")

        context = self._dataset_builder.build_context(settings)
        overrides = flexible_dataset_overrides(
            tuple(data.features or ()), tuple(data.targets or ())
        )
        dataset = self._dataset_builder.build_dataset(settings, context, overrides)
        datamodule, split_artifact = _build_datamodule(settings, self._dataset_builder, dataset)

        model_settings = with_runtime_module_defaults(settings.model)
        if model_settings is None:
            raise ValueError("MODEL settings are required but not configured")
        model, _ = _build_model_from_settings(model_settings)

        return RuntimeComponents(
            model=cast(LightningModule, model),
            datamodule=datamodule,
            trainer=None,
            artifacts=RuntimeArtifactManifest(split_artifact=split_artifact),
            meta={"dataset_type": "fit"},
        )
