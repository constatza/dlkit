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
from torch.nn import Identity

from dlkit.core.models.wrappers.components import (
    MetricRoute,
    ModelOutputSpec,
    NamedBatchTransformer,
    RoutedLossComputer,
    RoutedMetricsUpdater,
    WrapperCheckpointMetadata,
    _build_invoker_from_entries,
    _classify_feature_entries,
)
from dlkit.core.models.wrappers.protocols import IFittableBatchTransformer
from dlkit.core.training.transforms.chain import TransformChain
from dlkit.tools.config import (
    BuildContext,
    FactoryProvider,
    ModelComponentSettings,
    WrapperComponentSettings,
)
from dlkit.tools.config.components.model_components import LossInputRef
from dlkit.tools.config.data_entries import DataEntry, is_feature_entry, is_target_entry

from .base import ProcessingLightningWrapper, _build_model_from_settings
from .prediction_strategies import DiscriminativePredictionStrategy

if TYPE_CHECKING:
    from dlkit.core.shape_specs.simple_inference import ShapeSummary


def _make_chain(entry: DataEntry) -> Any:
    """Create a TransformChain for an entry, or Identity if no transforms configured.

    Args:
        entry: Data entry configuration with optional transforms attribute.

    Returns:
        TransformChain if transforms configured, nn.Identity otherwise.
    """
    settings = getattr(entry, "transforms", None)
    return (
        TransformChain(settings, entry_name=getattr(entry, "name", None))
        if settings
        else Identity()
    )


def _make_routes(
    metric_specs: tuple,
    default_target_key: str,
) -> list[MetricRoute]:
    """Build MetricRoute list from metric settings.

    Args:
        metric_specs: Tuple of MetricComponentSettings.
        default_target_key: Target name used when spec.target_key is None.

    Returns:
        List of MetricRoute value objects.
    """
    routes = []
    for spec in metric_specs:
        metric = FactoryProvider.create_component(spec, BuildContext(mode="training"))
        target_key_str = getattr(spec, "target_key", None)
        if target_key_str:
            target_name = target_key_str.split(".", 1)[1]
        else:
            target_name = default_target_key
        extra_inputs = getattr(spec, "extra_inputs", ()) or ()
        routes.append(
            MetricRoute(
                metric=metric,
                target_ns="targets",
                target_name=target_name,
                extra_inputs=tuple(extra_inputs),
            )
        )
    return routes


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


def _build_auto_extra_inputs(
    entry_configs: tuple[DataEntry, ...],
) -> dict[str, LossInputRef]:
    """Derive LossInputRef entries from DataEntry objects that declare loss_input.

    Each entry with a non-None loss_input value is auto-routed as a loss function
    kwarg. The kwarg name is the loss_input string; the batch key is derived from
    the entry's namespace and name.

    Args:
        entry_configs: Tuple of DataEntry objects in config-insertion order.

    Returns:
        Dict mapping kwarg name to LossInputRef, ready to merge with explicit routes.

    Raises:
        ValueError: If two entries declare the same loss_input kwarg name.
    """
    result: dict[str, LossInputRef] = {}
    for e in entry_configs:
        kwarg = getattr(e, "loss_input", None)
        if kwarg is None or e.name is None:
            continue
        if kwarg in result:
            raise ValueError(
                f"Duplicate loss_input kwarg '{kwarg}' declared on multiple entries. "
                f"Each kwarg name must appear on exactly one entry."
            )
        namespace = "features" if is_feature_entry(e) else "targets"
        result[kwarg] = LossInputRef(arg=kwarg, key=f"{namespace}.{e.name}")
    return result


def _merge_extra_inputs(
    auto: dict[str, LossInputRef],
    explicit: tuple[LossInputRef, ...],
) -> tuple[LossInputRef, ...]:
    """Merge auto-derived and explicit LossInputRef collections.

    Any overlap between auto-derived (from DataEntry.loss_input) and explicit
    (from LossComponentSettings.extra_inputs) routes is a configuration error —
    no silent overrides.

    Args:
        auto: Auto-derived routes keyed by kwarg name.
        explicit: Explicitly configured routes from LossComponentSettings.

    Returns:
        Merged tuple of LossInputRef with no duplicate arg names.

    Raises:
        ValueError: If the same kwarg name appears in both auto and explicit routes.
    """
    explicit_by_arg = {r.arg: r for r in explicit}
    overlap = set(auto) & set(explicit_by_arg)
    if overlap:
        raise ValueError(
            f"Loss kwarg(s) {sorted(overlap)} declared on both DataEntry.loss_input and "
            f"LossComponentSettings.extra_inputs. Remove one declaration — no silent overrides."
        )
    return tuple({**auto, **explicit_by_arg}.values())


def _ordered_model_input_names(feature_entries: list[DataEntry]) -> tuple[str, ...]:
    """Return feature entry names in invoker dispatch order.

    Mirrors :func:`_build_invoker_from_entries` ordering: positionals first
    (sorted by index), then kwargs (entry insertion order).  Used to store
    ``feature_names`` in checkpoint metadata so inference can map positional
    tensor args to the correct transform chain.

    Args:
        feature_entries: Feature DataEntry objects in config-insertion order.

    Returns:
        Tuple of entry names in the same order that the invoker dispatches
        tensors to ``model.forward()``.
    """
    positional, kwarg_map = _classify_feature_entries(feature_entries)
    positional.sort(key=lambda x: x[0])
    return tuple(name for _, name in positional) + tuple(kwarg_map.keys())


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
        **kwargs: Any,
    ) -> None:
        """Build protocols from settings and initialise the base wrapper.

        Args:
            settings: Wrapper configuration (loss, metrics, optimizer, scheduler).
            model_settings: Model configuration for building the nn.Module.
            entry_configs: Data entry configurations in config-insertion order.
            shape_summary: Shape summary from dataset inference (for shape-aware models).
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

        # --- Build transform chains ---
        feature_chains: dict[str, Any] = {
            e.name: _make_chain(e) for e in feature_entries if e.name is not None
        }
        target_chains: dict[str, Any] = {
            e.name: _make_chain(e) for e in target_entries if e.name is not None
        }
        batch_transformer = NamedBatchTransformer(feature_chains, target_chains)

        # --- Build model invoker (resolves model_input ordering) ---
        output_spec = ModelOutputSpec()
        model_invoker = _build_invoker_from_entries(feature_entries, output_spec)

        # --- Build loss computer ---
        loss_fn = FactoryProvider.create_component(
            settings.loss_function, BuildContext(mode="training")
        )
        loss_spec = settings.loss_function
        auto_extra = _build_auto_extra_inputs(entry_configs)
        explicit_extra = tuple(getattr(loss_spec, "extra_inputs", ()) or ())
        merged_extra = _merge_extra_inputs(auto_extra, explicit_extra)
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
        metric_specs = tuple(getattr(settings, "metrics", ()) or ())
        metrics_updater = RoutedMetricsUpdater(
            val_routes=_make_routes(metric_specs, default_target_key),
            test_routes=_make_routes(metric_specs, default_target_key),
        )

        # --- Build checkpoint metadata ---
        checkpoint_metadata = WrapperCheckpointMetadata(
            model_settings=model_settings,
            wrapper_settings=settings,
            entry_configs=entry_configs,
            feature_names=_ordered_model_input_names(feature_entries),
            predict_target_key=predict_target_key,
            shape_summary=shape_summary,
            output_spec=output_spec,
        )

        prediction_strategy = DiscriminativePredictionStrategy(
            model_invoker=model_invoker,
            batch_transformer=batch_transformer,
            predict_target_key=predict_target_key,
        )

        super().__init__(
            model=model,
            model_invoker=model_invoker,
            loss_computer=loss_computer,
            metrics_updater=metrics_updater,
            batch_transformer=batch_transformer,
            optimizer_settings=settings.optimizer,
            scheduler_settings=getattr(settings, "scheduler", None),
            predict_target_key=predict_target_key,
            checkpoint_metadata=checkpoint_metadata,
            prediction_strategy=prediction_strategy,
        )

        # Store protocol objects for use in _run_step and other methods
        self._model_invoker = model_invoker
        self._loss_computer = loss_computer
        self._batch_transformer = batch_transformer
        self._prediction_strategy = prediction_strategy
        self._train_generator_factory = self._train_generator_factory
        self._val_generator_factory = self._val_generator_factory

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
        from dlkit.core.models.wrappers.callbacks import TransformFittingCallback

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
        from dlkit.core.models.wrappers.base import _batch_size_of

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
