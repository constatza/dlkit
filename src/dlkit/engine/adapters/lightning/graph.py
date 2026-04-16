"""Graph Lightning wrapper for PyTorch Geometric models.

Accepts PyG Data/Batch objects directly; bypasses TensorDict entirely.
All step methods are overridden so the base protocol objects are never invoked
for the graph data path.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.data import Data
from torchmetrics import MetricCollection

from dlkit.engine.adapters.lightning.transform_pipeline import NamedBatchTransformer
from dlkit.engine.adapters.lightning.wrapper_types import (
    WrapperCheckpointMetadata,
    WrapperComponents,
)
from dlkit.infrastructure.config import (
    ModelComponentSettings,
    WrapperComponentSettings,
)
from dlkit.infrastructure.config.data_entries import DataEntry, is_feature_entry, is_target_entry

from .base import ProcessingLightningWrapper, _build_model_from_settings
from .prediction_strategies import DiscriminativePredictionStrategy


class GraphLightningWrapper(ProcessingLightningWrapper):
    """Lightning wrapper for PyTorch Geometric graph neural networks.

    Accepts PyG ``Data``/``Batch`` objects from a PyG DataLoader directly —
    no TensorDict conversion needed. Target is extracted by the first
    target-entry config name (default ``"y"``). All Lightning step methods
    are overridden; the base protocol objects serve as no-op sentinels.

    **batch_size semantics**: All step methods pass ``batch_size=predictions.shape[0]``
    (total node count M across the PyG batch) to ``self.log()``. This is
    mathematically correct for node-level losses with mean reduction: Lightning
    computes the epoch mean as ``Σ(loss_i × M_i) / Σ(M_i)``, recovering the
    true per-node mean over the full dataset. Using ``num_graphs`` instead
    would bias the epoch mean toward batches with more nodes per graph, and
    it is also unavailable on single ``Data`` objects.

    Attributes:
        loss_function: Loss callable for training/validation/test steps.
        val_metrics: Validation MetricCollection.
        test_metrics: Test MetricCollection.
    """

    def __init__(
        self,
        *,
        settings: WrapperComponentSettings,
        model_settings: ModelComponentSettings,
        entry_configs: tuple[DataEntry, ...] | None = None,
        shape_summary: Any = None,
        components: WrapperComponents,
        **kwargs: Any,
    ) -> None:
        """Initialise the graph wrapper.

        Wires pre-built components and null protocol sentinels into the base class
        (the step methods below override everything so the sentinels are never invoked).

        Args:
            settings: Wrapper configuration (loss, metrics, optimizer, scheduler).
            model_settings: Model configuration for building the nn.Module.
            entry_configs: Data entry configurations.
            shape_summary: Shape summary from dataset inference (optional).
            components: Pre-built WrapperComponents containing loss, metrics, transforms,
                optimizer factory, and scheduler factory.
            **kwargs: Forwarded to LightningModule.
        """
        entry_configs = entry_configs or ()

        # Build model and value objects before calling super().__init__()
        model = _build_model_from_settings(model_settings, shape_summary)

        # Use injected components
        _loss_function = components.loss_fn
        _val_metrics = MetricCollection([r.metric for r in components.val_metric_routes])
        _test_metrics = MetricCollection([r.metric for r in components.test_metric_routes])

        feature_entries = [e for e in entry_configs if is_feature_entry(e)]
        checkpoint_metadata = WrapperCheckpointMetadata(
            model_settings=model_settings,
            wrapper_settings=settings,
            entry_configs=entry_configs,
            feature_names=tuple(e.name for e in feature_entries if e.name is not None),
            predict_target_key="",
            shape_summary=shape_summary,
        )

        # Create minimal protocol objects (graph overrides all step methods)
        from dlkit.engine.adapters.lightning.null_components import (
            _NullLossComputer,
            _NullMetricsUpdater,
            _NullModelInvoker,
        )

        _null_transformer = NamedBatchTransformer({}, {})
        _null_invoker = _NullModelInvoker()
        _prediction_strategy = DiscriminativePredictionStrategy(
            model_invoker=_null_invoker,
            batch_transformer=_null_transformer,
            predict_target_key="",
        )

        # Build optimization controller for graph models
        from dlkit.engine.training.optimization.builder import OptimizationProgramBuilder
        from dlkit.engine.training.optimization.controllers import (
            AutomaticOptimizationController,
            ManualOptimizationController,
        )
        from dlkit.engine.training.optimization.state_repository import OptimizationStateRepository
        from dlkit.engine.training.optimization.stepping import StepAllOptimizers

        program = OptimizationProgramBuilder().build(
            model, components.optimization_program_settings
        )
        repository = OptimizationStateRepository()

        # Determine if manual optimization is required
        def _requires_manual(program: Any) -> bool:
            """Check if any stage uses LBFGS-like optimizer requiring manual stepping."""
            from dlkit.engine.training.optimization.state import ActiveStage

            for entry in program.stages:
                # Check if entry is ActiveStage or ActiveConcurrentGroup
                if isinstance(entry, ActiveStage):
                    stages = [entry]
                else:
                    stages = entry.stages
                for stage in stages:
                    opt_cls = stage.optimizer.__class__.__name__.lower()
                    if any(x in opt_cls for x in ["lbfgs", "manual"]):
                        return True
            return False

        if _requires_manual(program):
            optimization_controller = ManualOptimizationController(
                program, repository, StepAllOptimizers()
            )
        else:
            optimization_controller = AutomaticOptimizationController(program, repository)

        # super().__init__() must be called before assigning any nn.Module attributes
        super().__init__(
            model=model,
            model_invoker=_null_invoker,
            loss_computer=_NullLossComputer(),
            metrics_updater=_NullMetricsUpdater(),
            batch_transformer=_null_transformer,
            optimization_controller=optimization_controller,
            predict_target_key="",
            checkpoint_metadata=checkpoint_metadata,
            prediction_strategy=_prediction_strategy,
        )

        # Assign nn.Module attributes AFTER super().__init__()
        self.val_metrics = _val_metrics
        self.test_metrics = _test_metrics
        self.loss_function = _loss_function

        # Resolved once at init from config; never referenced in the data hot path
        self._graph_target_name: str = next(
            (e.name for e in entry_configs if is_target_entry(e) and e.name is not None),
            "y",
        )

    @staticmethod
    def _decompose_pyg_batch(
        batch: Data | PyGBatch,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Extract (x, edge_index, edge_attr) from a PyG Data/Batch.

        Args:
            batch: PyG Data or Batch object to decompose.

        Returns:
            Tuple of (x, edge_index, edge_attr) tensors.
        """
        graph_batch = cast(Any, batch)
        x: Tensor = cast(Tensor, graph_batch.x)
        edge_index: Tensor = cast(Tensor, graph_batch.edge_index)
        edge_attr: Tensor | None = cast(Tensor | None, getattr(graph_batch, "edge_attr", None))
        if edge_attr is None:
            edge_attr = cast(Tensor | None, getattr(graph_batch, "edge_weight", None))
        return x, edge_index, edge_attr

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through the model with decomposed graph tensors.

        Args:
            x: Node feature tensor.
            edge_index: Edge connectivity tensor (2 × num_edges).
            edge_attr: Optional edge attribute tensor.

        Returns:
            Model output tensor.
        """
        return self.model(x, edge_index, edge_attr)

    # =========================================================================
    # Template Method Implementation
    # =========================================================================

    def _run_step(self, batch: Any, batch_idx: int, stage: str) -> tuple[Tensor, int | None, Any]:
        """Execute one forward+loss step for graph data.

        Graph data is not enriched into a TensorDict; this method is provided
        for API compatibility but should never be called since all step methods
        are overridden.

        Args:
            batch: PyG Data or Batch (not used).
            batch_idx: Batch index (not used).
            stage: Stage identifier (not used).

        Raises:
            NotImplementedError: Always, since this should not be called.
        """
        raise NotImplementedError(
            "GraphLightningWrapper overrides all step methods directly. "
            "_run_step should never be called."
        )

    # =========================================================================
    # Step overrides: PyG Data/Batch → model directly
    # =========================================================================

    def training_step(self, batch: Data | PyGBatch, batch_idx: int) -> dict[str, Any]:
        """Training step for PyG Data/Batch.

        Args:
            batch: PyG Data or Batch from a PyG DataLoader.
            batch_idx: Index of the batch.

        Returns:
            Dictionary containing the training loss.

        Note:
            ``batch_size=predictions.shape[0]`` passes the total node count so
            Lightning weights epoch-level aggregation by nodes, not graphs.
        """
        x, edge_index, edge_attr = self._decompose_pyg_batch(batch)
        predictions = self.model(x, edge_index, edge_attr)
        target = self._extract_pyg_target(batch, predictions)
        loss = self.loss_function(predictions, target)
        self._log_stage_outputs("train", loss, batch_size=predictions.shape[0])
        return {"loss": loss}

    def validation_step(self, batch: Data | PyGBatch, batch_idx: int) -> dict[str, Any]:
        """Validation step for PyG Data/Batch.

        Args:
            batch: PyG Data or Batch from a PyG DataLoader.
            batch_idx: Index of the batch.

        Returns:
            Dictionary containing the validation loss.

        Note:
            ``batch_size=predictions.shape[0]`` passes the total node count so
            Lightning weights epoch-level aggregation by nodes, not graphs.
        """
        x, edge_index, edge_attr = self._decompose_pyg_batch(batch)
        predictions = self.model(x, edge_index, edge_attr)
        target = self._extract_pyg_target(batch, predictions)
        val_loss = self.loss_function(predictions, target)
        metrics = self.val_metrics(predictions, target)
        self._log_stage_outputs(
            "val", val_loss, metrics if metrics else None, batch_size=predictions.shape[0]
        )
        return {"val_loss": val_loss}

    def test_step(self, batch: Data | PyGBatch, batch_idx: int) -> dict[str, Any]:
        """Test step for PyG Data/Batch.

        Args:
            batch: PyG Data or Batch from a PyG DataLoader.
            batch_idx: Index of the batch.

        Returns:
            Dictionary containing the test loss.

        Note:
            ``batch_size=predictions.shape[0]`` passes the total node count so
            Lightning weights epoch-level aggregation by nodes, not graphs.
        """
        x, edge_index, edge_attr = self._decompose_pyg_batch(batch)
        predictions = self.model(x, edge_index, edge_attr)
        target = self._extract_pyg_target(batch, predictions)
        test_loss = self.loss_function(predictions, target)
        metrics = self.test_metrics(predictions, target)
        self._log_stage_outputs(
            "test", test_loss, metrics if metrics else None, batch_size=predictions.shape[0]
        )
        return {"test_loss": test_loss}

    def predict_step(self, batch: Data | PyGBatch, batch_idx: int) -> Any:
        """Predict step for PyG Data/Batch returning a TensorDict.

        Always emits all three keys (``"predictions"``, ``"targets"``,
        ``"latents"``) using zero-size sentinels when a field is absent, so
        that ``torch.cat`` across batches is unconditionally safe.

        Args:
            batch: PyG Data or Batch from a PyG DataLoader.
            batch_idx: Index of the batch.

        Returns:
            TensorDict with keys ``"predictions"``, ``"targets"``, and
            ``"latents"`` (zero-size sentinels when absent).
        """
        from tensordict import TensorDict

        raw_target: Tensor | None = getattr(batch, self._graph_target_name, None)
        x, edge_index, edge_attr = self._decompose_pyg_batch(batch)
        predictions = self.model(x, edge_index, edge_attr)
        batch_size = predictions.shape[0]
        sentinel = torch.zeros(batch_size, 0, dtype=predictions.dtype, device=predictions.device)
        return TensorDict(
            {
                "predictions": predictions,
                "targets": raw_target if raw_target is not None else sentinel,
                "latents": sentinel,
            },
            batch_size=predictions.shape[:1],
        )

    def on_validation_epoch_end(self) -> None:
        """Reset validation metrics at the end of the epoch."""
        self.val_metrics.reset()

    def on_test_epoch_end(self) -> None:
        """Reset test metrics at the end of the epoch."""
        self.test_metrics.reset()

    def _extract_pyg_target(self, batch: Data | PyGBatch, predictions: Tensor) -> Tensor:
        """Extract target from PyG batch and align dtype to predictions.

        Args:
            batch: PyG Data or Batch with a target attribute.
            predictions: Model predictions tensor (used for dtype alignment).

        Returns:
            Target tensor with dtype matching predictions.

        Raises:
            AttributeError: If the target attribute is not found on the batch.
        """
        target: Tensor = getattr(batch, self._graph_target_name)
        if target.is_floating_point() and target.dtype != predictions.dtype:
            target = target.to(dtype=predictions.dtype)
        return target
