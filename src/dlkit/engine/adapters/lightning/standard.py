"""Standard Lightning wrapper for tensor/TensorDict-based models.

Translates the settings-based API (settings, model_settings, entry_configs) into
protocol objects and delegates to ProcessingLightningWrapper. This keeps the
external constructor API stable while the base class uses pure protocol injection.
"""

from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from dlkit.engine.adapters.lightning.loss_routing import (
    RoutedLossComputer,
    build_auto_extra_inputs,
    merge_extra_inputs,
)
from dlkit.engine.adapters.lightning.metrics_routing import RoutedMetricsUpdater
from dlkit.engine.adapters.lightning.model_invoker import (
    ModelOutputSpec,
    _build_invoker_from_entries,
)
from dlkit.engine.adapters.lightning.protocols import IFittableBatchTransformer
from dlkit.engine.adapters.lightning.transform_pipeline import build_batch_transformer
from dlkit.engine.adapters.lightning.wrapper_types import (
    WrapperComponents,
    build_checkpoint_metadata,
)
from dlkit.engine.training.optimization.controllers import build_optimization_controller
from dlkit.infrastructure.config import (
    ModelComponentSettings,
    WrapperComponentSettings,
)
from dlkit.infrastructure.config.data_entries import DataEntry, is_feature_entry, is_target_entry
from dlkit.infrastructure.config.model_components import LossInputRef

from .base import ProcessingLightningWrapper, _build_model_from_settings
from .prediction_strategies import DiscriminativePredictionStrategy

if TYPE_CHECKING:
    from dlkit.common.shapes import ShapeSummary


def _validate_extra_inputs_against_signature(
    loss_fn: Any,
    extra_inputs: tuple[LossInputRef, ...],
) -> None:
    """Best-effort build-time check that extra_inputs kwargs exist in the loss function.

    Inspects the loss function signature and raises ValueError if a routed kwarg name
    is not present and the function does not accept **kwargs. Skips validation for
    uninspectable callables (e.g. C extensions, lambdas with *args).

    Args:
        loss_fn: The loss function to validate against.
        extra_inputs: Merged extra input refs to validate.

    Raises:
        ValueError: If a kwarg name is not accepted by the loss function.
    """
    if not extra_inputs:
        return
    import inspect

    try:
        sig = inspect.signature(loss_fn)
    except ValueError, TypeError:
        return  # Uninspectable — skip validation
    params = sig.parameters
    accepts_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_var_keyword:
        return  # **kwargs — all kwarg names are valid
    positional_only = {
        name for name, p in params.items() if p.kind == inspect.Parameter.POSITIONAL_ONLY
    }
    for ref in extra_inputs:
        if ref.arg not in params:
            available = [n for n in params if n not in positional_only]
            raise ValueError(
                f"Loss function has no parameter named '{ref.arg}'. "
                f"Available non-positional-only parameters: {available}. "
                f"Check DataEntry.loss_input or LossComponentSettings.extra_inputs."
            )
        if ref.arg in positional_only:
            raise ValueError(
                f"Loss function parameter '{ref.arg}' is positional-only (declared with '/') "
                f"and cannot be passed as a keyword argument. "
                f"Rename the parameter or adjust loss routing."
            )


class StandardLightningWrapper(ProcessingLightningWrapper):
    """Concrete wrapper for tensor/TensorDict-based models.

    Translates settings-based constructor arguments into protocol objects and
    passes them to the base class. Supports named transform chains via
    NamedBatchTransformer.

    Example:
        ```python
        wrapper = StandardLightningWrapper(
            settings=wrapper_settings,
            model_settings=model_settings,
            entry_configs=data_configs,
            shape_summary=shape_summary,
        )
        ```
    """

    def __init__(
        self,
        *,
        settings: WrapperComponentSettings,
        model_settings: ModelComponentSettings,
        entry_configs: tuple[DataEntry, ...] | None = None,
        shape_summary: ShapeSummary | None = None,
        components: WrapperComponents,
        **kwargs: Any,
    ) -> None:
        """Build protocols from settings and initialise the base wrapper.

        Args:
            settings: Wrapper configuration (loss, metrics, optimizer, scheduler).
            model_settings: Model configuration for building the nn.Module.
            entry_configs: Data entry configurations in config-insertion order.
            shape_summary: Shape summary from dataset inference (for shape-aware models).
            components: Pre-built WrapperComponents containing loss, metrics, transforms,
                optimizer factory, and scheduler factory.
            **kwargs: Forwarded to LightningModule (ignored otherwise).
        """
        entry_configs = entry_configs or ()

        # --- Build model ---
        model = _build_model_from_settings(model_settings, shape_summary)

        # --- Partition entries ---
        feature_entries = [e for e in entry_configs if is_feature_entry(e)]
        target_entries = [e for e in entry_configs if is_target_entry(e)]

        all_target_keys: tuple[str, ...] = tuple(
            e.name for e in target_entries if e.name is not None
        )
        default_target_key: str = all_target_keys[0] if all_target_keys else ""

        # --- Use injected components ---
        loss_fn = components.loss_fn
        val_metric_routes = components.val_metric_routes
        test_metric_routes = components.test_metric_routes

        # --- Build batch transformer ---
        batch_transformer = build_batch_transformer(
            components.feature_transforms, components.target_transforms
        )

        # --- Build model invoker (resolves model_input ordering) ---
        output_spec = ModelOutputSpec()
        model_invoker = _build_invoker_from_entries(feature_entries, output_spec)

        # --- Build loss computer ---
        loss_spec = settings.loss_function
        auto_extra = build_auto_extra_inputs(entry_configs)
        explicit_extra = tuple(getattr(loss_spec, "extra_inputs", ()) or ())
        merged_extra = merge_extra_inputs(auto_extra, explicit_extra)
        _validate_extra_inputs_against_signature(loss_fn, merged_extra)
        loss_computer = RoutedLossComputer(
            loss_fn=loss_fn,
            target_key=getattr(loss_spec, "target_key", None),
            default_target_key=default_target_key,
            extra_inputs=merged_extra,
        )

        # --- Derive predict target key ---
        loss_target_key: str | None = getattr(loss_spec, "target_key", None)
        predict_target_key: str = (
            loss_target_key.split(".", 1)[1] if loss_target_key else default_target_key
        )

        # --- Build metrics updater ---
        metrics_updater = RoutedMetricsUpdater(
            val_routes=val_metric_routes,
            test_routes=test_metric_routes,
        )

        # --- Build checkpoint metadata ---
        checkpoint_metadata = build_checkpoint_metadata(
            model_settings=model_settings,
            wrapper_settings=settings,
            entry_configs=entry_configs,
            feature_entries=feature_entries,
            predict_target_key=predict_target_key,
            shape_summary=shape_summary,
            output_spec=output_spec,
        )

        prediction_strategy = DiscriminativePredictionStrategy(
            model_invoker=model_invoker,
            batch_transformer=batch_transformer,
            predict_target_key=predict_target_key,
        )

        # Build optimization controller
        optimization_controller = build_optimization_controller(
            model, components.optimizer_policy_settings
        )

        super().__init__(
            model=model,
            model_invoker=model_invoker,
            loss_computer=loss_computer,
            metrics_updater=metrics_updater,
            batch_transformer=batch_transformer,
            optimization_controller=optimization_controller,
            predict_target_key=predict_target_key,
            checkpoint_metadata=checkpoint_metadata,
            prediction_strategy=prediction_strategy,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor.

        Returns:
            Model output tensor.
        """
        return self.model(x)

    # =========================================================================
    # Standard-Specific Hooks and Methods
    # =========================================================================

    def configure_callbacks(self) -> list[Any]:
        """Register lifecycle callbacks for this wrapper.

        Returns a ``TransformFittingCallback`` when the batch transformer
        implements ``IFittableBatchTransformer``, so Lightning automatically
        fits transforms before the first training epoch.

        Returns:
            List of Lightning Callbacks to attach to the Trainer.
        """
        from dlkit.engine.adapters.lightning.callbacks import TransformFittingCallback

        if isinstance(self._batch_transformer, IFittableBatchTransformer):
            return [TransformFittingCallback(self._batch_transformer)]
        return []

    def _apply_batch_transforms(
        self,
        batch: Any,
        generator: torch.Generator | None,
    ) -> Any:
        """Apply coupled supervision transforms in sequence.

        Each ``IBatchTransform`` in ``self._batch_transforms`` is called with
        the batch and generator, the result piped to the next transform via
        ``reduce``.  An empty sequence is a no-op.

        Args:
            batch: Input TensorDict.
            generator: Optional RNG for stochastic transforms.

        Returns:
            Transformed TensorDict (identity when no transforms configured).
        """
        if not self._batch_transforms:
            return batch
        return reduce(lambda b, t: t(b, generator), self._batch_transforms, batch)

    def _compute_loss(self, predictions: Any, batch: Any) -> Tensor:
        """Compute scalar loss from predictions and batch.

        Delegates to ``self._loss_computer`` by default. Subclasses can
        override this method to implement custom loss logic (e.g.
        ``FlowMatchingWrapper`` targets ``batch["targets"]["ut"]`` directly).

        Args:
            predictions: Model output tensor.
            batch: Enriched TensorDict with features, targets, and predictions.

        Returns:
            Scalar loss tensor.
        """
        return self._loss_computer.compute(predictions, batch)

    # =========================================================================
    # Template Method Implementation
    # =========================================================================

    def _run_step(self, batch: Any, batch_idx: int, stage: str) -> tuple[Tensor, int | None, Any]:
        """Execute one forward+loss step.

        Implements the template method from the base class. Applies batch transforms,
        applies per-slot transforms, invokes the model, and computes loss.

        Args:
            batch: Input batch from dataset.
            batch_idx: Index of the batch.
            stage: Stage identifier ('train', 'val', 'test').

        Returns:
            Tuple of (loss, batch_size, enriched_batch).
        """
        from dlkit.engine.adapters.lightning.base import _batch_size_of

        gen = (self._train_generator_factory if stage == "train" else self._val_generator_factory)(
            batch_idx
        )
        batch = self._apply_batch_transforms(batch, gen)
        batch = self._batch_transformer.transform(batch)
        batch = self._model_invoker.invoke(self.model, batch)
        loss = self._compute_loss(batch["predictions"], batch)
        batch_size = _batch_size_of(batch["predictions"])
        return loss, batch_size, batch

    # =========================================================================
    # Checkpoint Customization
    # =========================================================================

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Augment checkpoint with target_names metadata.

        Args:
            checkpoint: Checkpoint dict to augment.
        """
        super().on_save_checkpoint(checkpoint)
        if "dlkit_metadata" in checkpoint:
            meta = self._checkpoint_metadata
            if meta is not None:
                target_names = [
                    e.name for e in meta.entry_configs if is_target_entry(e) and e.name is not None
                ]
                checkpoint["dlkit_metadata"]["target_names"] = target_names
