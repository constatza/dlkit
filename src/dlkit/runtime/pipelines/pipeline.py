"""Processing pipeline implementing Chain of Responsibility pattern.

This module defines a flexible processing pipeline where each step in the chain
handles a specific aspect of dataflow processing. The pipeline follows the Template
Method pattern with the Chain of Responsibility pattern for step execution.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

import torch
from loguru import logger

from dlkit.core.training.transforms.chain import TransformChain
from dlkit.core.training.transforms.interfaces import IFittableTransform
from dlkit.tools.config.data_entries import DataEntry, Feature, Target, is_feature_entry, is_target_entry
from .context import ProcessingContext
from .interfaces import ModelInvoker, OutputClassifier, OutputNamer


class ProcessingStep(ABC):
    """Abstract base class for processing steps using Chain of Responsibility.

    Each processing step handles one specific aspect of dataflow processing and
    can optionally pass control to the next step in the chain. This follows
    the Template Method pattern where the framework defines the control flow
    but delegates specific processing to subclasses.

    Attributes:
        _next_step: The next step in the processing chain
    """

    def __init__(self, next_step: ProcessingStep = None):
        """Initialize the processing step.

        Args:
            next_step (ProcessingStep, optional): The next step in the processing chain.
        """
        self._next_step = next_step

    def handle(self, context: ProcessingContext) -> ProcessingContext:
        """Run this step and optionally the next step.

        This implements the Template Method pattern: subclasses implement
        ``process`` and this method chains the steps.

        Args:
            context (ProcessingContext): Processing context carrying dataflow through the pipeline.

        Returns:
            ProcessingContext: Updated processing context.
        """
        # Process current step
        context = self.process(context)

        # Continue to next step if available
        if self._next_step:
            context = self._next_step.handle(context)

        return context

    @abstractmethod
    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Process the context at this step.

        Args:
            context (ProcessingContext): Processing context to modify.

        Returns:
            ProcessingContext: Updated processing context.
        """
        pass

    def set_next(self, next_step: ProcessingStep) -> None:
        """Set the next step in the processing chain.

        Args:
            next_step (ProcessingStep): The next processing step.
        """
        self._next_step = next_step


class DataExtractionStep(ProcessingStep):
    """Extract and categorize dataflow from raw batch into features and targets.

    This step separates the raw batch dataflow into features (model inputs)
    and targets (ground truth dataflow) based on the dataflow entry configurations.

    Attributes:
        _entry_configs: Dictionary mapping dataflow names to their configurations
        _feature_names: Set of feature names for quick lookup
        _target_names: Set of target names for quick lookup
    """

    def __init__(self, entry_configs: dict[str, DataEntry], next_step: ProcessingStep = None):
        """Initialize the dataflow extraction step.

        Args:
            entry_configs (dict[str, DataEntry]): Mapping from entry name to configuration.
            next_step (ProcessingStep | None): Next step in the processing chain.
        """
        super().__init__(next_step)
        self._entry_configs = entry_configs

        # Pre-compute name sets for efficient lookup
        self._feature_names = {
            name for name, config in entry_configs.items() if is_feature_entry(config)
        }
        self._target_names = {
            name for name, config in entry_configs.items() if is_target_entry(config)
        }

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Extract features and targets from raw batch.

        When no explicit entry configuration is provided, apply a simple
        name-based heuristic to separate features and targets:
        - targets: keys like {"y", "target", "targets", "label", "labels"} (case-insensitive)
        - features: all remaining keys

        Args:
            context (ProcessingContext): Processing context with ``raw_batch``

        Returns:
            ProcessingContext: Context with populated ``features`` and ``targets``.
        """
        if not self._feature_names and not self._target_names:
            # Heuristic fallback when no configs are provided
            target_like = {"y", "target", "targets", "label", "labels"}
            features: dict[str, torch.Tensor] = {}
            targets: dict[str, torch.Tensor] = {}
            for name, tensor in context.raw_batch.items():
                n = str(name).lower()
                if n in target_like:
                    targets[name] = tensor
                else:
                    features[name] = tensor
            context.features = features
            context.targets = targets
            return context

        # Extract based on explicit configuration
        context.features = {
            name: tensor
            for name, tensor in context.raw_batch.items()
            if name in self._feature_names
        }
        context.targets = {
            name: tensor for name, tensor in context.raw_batch.items() if name in self._target_names
        }
        return context


class PrecisionValidationStep(ProcessingStep):
    """Validate that feature tensors match the model's precision."""

    def __init__(
        self,
        model_invoker: ModelInvoker,
        next_step: ProcessingStep | None = None,
    ):
        super().__init__(next_step)
        self._model_invoker = model_invoker

    def process(self, context: ProcessingContext) -> ProcessingContext:
        model = getattr(self._model_invoker, "model", None)
        model_dtype = getattr(model, "dtype", None)

        if model_dtype is None:
            return context

        mismatches: list[str] = []
        for name, tensor in context.features.items():
            if tensor.is_floating_point() and tensor.dtype != model_dtype:
                mismatches.append(f"{name}={tensor.dtype}")

        if mismatches:
            mismatch_details = ", ".join(mismatches)
            message = (
                "Feature tensors have precision mismatches "
                f"(expected {model_dtype}, got {mismatch_details})"
            )
            logger.error(message)
            raise RuntimeError(message)

        return context


class ModelInvocationStep(ProcessingStep):
    """Invoke the model with extracted features using Command Pattern.

    This step encapsulates the model invocation command, allowing different
    models to be invoked without the pipeline needing to know the specifics.

    Attributes:
        _model_invoker: Command object for model invocation
    """

    def __init__(self, model_invoker: ModelInvoker, next_step: ProcessingStep | None = None):
        """Initialize the model invocation step.

        Args:
            model_invoker (ModelInvoker): Command object for invoking the model.
            next_step (ProcessingStep | None): Next step in the processing chain.
        """
        super().__init__(next_step)
        self._model_invoker = model_invoker

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Invoke the model with features and store outputs.

        Performs defensive dtype validation to ensure features match the model's
        precision. Data should already be loaded in the correct dtype via load_array(),
        but this provides a safety check and automatic casting if needed.

        Args:
            context (ProcessingContext): Processing context with ``features`` set.

        Returns:
            ProcessingContext: Context with ``model_outputs`` populated.

        Raises:
            RuntimeError: If model invocation fails or no features are available.
        """
        if not context.features:
            raise RuntimeError("No features available for model invocation")

        try:
            # Debug-level dtype tracking (no casting - Lightning handles precision)
            # Lightning's precision plugin converts model parameters during trainer.fit()
            # Data is loaded with correct precision via PrecisionService
            # Temporary mismatches before Lightning setup are expected and harmless
            if hasattr(self._model_invoker.model, "dtype"):
                model_dtype = self._model_invoker.model.dtype
                for name, tensor in context.features.items():
                    if tensor.is_floating_point() and tensor.dtype != model_dtype:
                        logger.debug(
                            f"Feature '{name}' dtype {tensor.dtype} differs from current model dtype {model_dtype}. "
                            f"Lightning's precision plugin will handle alignment during forward pass."
                        )

            context.model_outputs = self._model_invoker.invoke(context.features)
        except Exception as e:
            raise RuntimeError(f"Model invocation step failed: {e}") from e

        return context


class OutputClassificationStep(ProcessingStep):
    """Classify model outputs into latents and predictions using Strategy Pattern.

    This step uses a configurable strategy to determine which model outputs
    are latent representations and which are predictions corresponding to targets.

    Attributes:
        _classifier: Strategy object for output classification
    """

    def __init__(self, classifier: OutputClassifier, next_step: ProcessingStep | None = None):
        """Initialize the output classification step.

        Args:
            classifier (OutputClassifier): Strategy object for classifying outputs.
            next_step (ProcessingStep | None): Next step in the processing chain.
        """
        super().__init__(next_step)
        self._classifier = classifier

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Classify model outputs into latents and predictions.

        Args:
            context (ProcessingContext): Processing context with ``model_outputs`` and ``targets``.

        Returns:
            ProcessingContext: Context with ``latents`` and ``predictions`` populated.

        Raises:
            RuntimeError: If output classification fails.
        """
        if not context.model_outputs:
            raise RuntimeError("No model outputs available for classification")

        try:
            context.latents, context.predictions = self._classifier.classify(
                context.model_outputs, context.targets
            )
        except Exception as e:
            raise RuntimeError(f"Output classification step failed: {e}") from e

        return context


class OutputNamingStep(ProcessingStep):
    """Rename prediction keys using a dedicated naming strategy.

    This step applies an ``OutputNamer`` to map prediction keys
    (typically model output names) to desired names, usually target keys.
    It runs after classification and before loss pairing/aggregation.
    """

    def __init__(self, namer: OutputNamer, next_step: ProcessingStep | None = None):
        super().__init__(next_step)
        self._namer = namer

    def process(self, context: ProcessingContext) -> ProcessingContext:
        if not context.predictions:
            return context
        try:
            context.predictions = self._namer.rename_predictions(
                context.predictions, context.targets, model_outputs=context.model_outputs
            )
        except Exception as e:
            raise RuntimeError(f"Output naming step failed: {e}") from e
        return context


class TransformApplicationStep(ProcessingStep):
    """Apply per-entry transform chains declared in dataflow entry configs.

    This step reads configured transforms from ``Feature``/``Target`` entries and
    applies them to the corresponding tensors in a batch via a ``TransformChain``.
    Chains are built once per entry name using the runtime input shape (including batch
    dimension) and cached for reuse across batches and phases.

    Args:
        entry_configs: Mapping of dataflow entry names to their configurations.
        next_step: Optional next processing step in the chain.
        cache: Optional shared cache mapping entry name to a pre-built ``TransformChain``.
            This allows train/val/test pipelines to share fitted chains.
        fit_during_apply: If True, calls ``chain.fit(x)`` the first time an entry is seen.
            Disable this after a global fit (e.g., in ``on_fit_start``) to avoid refitting.

    Example:
        Build a pipeline that extracts and transforms before model invocation::

            step = TransformApplicationStep(entry_configs)
            pipeline = ProcessingPipeline(DataExtractionStep(entry_configs, step))
            ctx = pipeline.execute(batch)  # ctx.features/targets transformed
    """

    def __init__(
        self,
        entry_configs: dict[str, DataEntry],
        next_step: ProcessingStep = None,
        *,
        cache: dict[str, TransformChain] | None = None,
        fit_during_apply: bool = True,
        wrapper: "ProcessingLightningWrapper | None" = None,
    ):
        super().__init__(next_step)
        self._entry_configs = entry_configs
        # Allow sharing cache across pipeline instances
        self._transform_cache: dict[str, TransformChain] = cache if cache is not None else {}
        # Control whether to call .fit() during apply
        self._enable_fit: bool = fit_during_apply
        # Allow disabling transform application entirely (runtime toggle)
        self._enabled: bool = True
        # Reference to wrapper for accessing fitted_transforms (single source of truth)
        self._wrapper = wrapper

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # Apply transforms to features and targets if configured
        if self._enabled:
            context.features = self._apply_group(context.features)
            context.targets = self._apply_group(context.targets)
        return context

    def _apply_group(self, group: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not group:
            return group
        result: dict[str, torch.Tensor] = {}
        for name, tensor in group.items():
            entry = self._entry_configs.get(name)
            if entry is None or not getattr(entry, "transforms", None):
                result[name] = tensor
                continue

            chain = self._get_or_build_transforms(name, entry, tensor)
            x = tensor
            try:
                if self._enable_fit and isinstance(chain, IFittableTransform):
                    chain.fit(x)
                x = chain(x)
            except Exception as e:
                raise RuntimeError(f"Transform chain failed for '{name}': {e}") from e
            result[name] = x
        return result

    def _get_or_build_transforms(
        self, name: str, entry: DataEntry, tensor: torch.Tensor
    ) -> TransformChain:
        """Get or build a transform chain for the given entry.

        Lookup order:
        1. Check local cache (for performance during training)
        2. Check wrapper.fitted_transforms (single source of truth for persistence)
        3. Create new chain (fallback with warnings)

        Args:
            name: Name of the entry (e.g., "features").
            entry: DataEntry configuration containing transform settings.
            tensor: Tensor to get shape from (for validation).

        Returns:
            TransformChain instance for the entry.
        """
        # First check local cache (fast path during training)
        cached = self._transform_cache.get(name)
        if cached is not None:
            return cached

        # Second, check wrapper's fitted transforms (loaded from checkpoint)
        # Try both feature and target transforms (separated by type)
        if self._wrapper is not None:
            fitted_chain = None

            # Check feature transforms first
            if hasattr(self._wrapper, 'fitted_feature_transforms'):
                try:
                    fitted_chain = self._wrapper.fitted_feature_transforms[name]
                    logger.debug(f"Using fitted feature transform chain for '{name}'")
                except KeyError:
                    pass

            # If not found in features, check target transforms
            if fitted_chain is None and hasattr(self._wrapper, 'fitted_target_transforms'):
                try:
                    fitted_chain = self._wrapper.fitted_target_transforms[name]
                    logger.debug(f"Using fitted target transform chain for '{name}'")
                except KeyError:
                    pass

            # If found in either, cache and return
            if fitted_chain is not None:
                self._transform_cache[name] = fitted_chain
                return fitted_chain

            # Not found in either - log available keys for debugging
            feature_keys = list(self._wrapper.fitted_feature_transforms.keys()) if hasattr(self._wrapper, 'fitted_feature_transforms') else []
            target_keys = list(self._wrapper.fitted_target_transforms.keys()) if hasattr(self._wrapper, 'fitted_target_transforms') else []
            logger.warning(
                f"Transform chain '{name}' not found in wrapper transforms. "
                f"Available: features={feature_keys}, targets={target_keys}"
            )

        # FALLBACK: Build new chain (should only happen during initial training)
        logger.warning(
            f"⚠️  CREATING NEW UNFITTED TRANSFORM CHAIN for '{name}' ⚠️\n"
            f"   This should only happen during initial training.\n"
            f"   If this occurs during inference/prediction after loading a checkpoint,\n"
            f"   it indicates a bug in transform persistence/restoration.\n"
            f"   Context: wrapper={self._wrapper is not None}, "
            f"cache_empty={len(self._transform_cache)==0}, "
            f"fitted_transforms={'present' if self._wrapper and hasattr(self._wrapper, 'fitted_transforms') else 'missing'}"
        )
        chain = TransformChain(entry.transforms)
        self._transform_cache[name] = chain
        return chain

    def set_fit_enabled(self, enabled: bool) -> None:
        """Enable or disable calling .fit() during transform application."""
        self._enable_fit = enabled

    @property
    def cache(self) -> dict[str, TransformChain]:
        return self._transform_cache

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable transform application entirely."""
        self._enabled = enabled


class LossDataAggregationStep(ProcessingStep):
    """Aggregate dataflow required for loss computation.

    This step collects all dataflow entries marked as required_in_loss from
    features, targets, latents, and predictions into a single loss_data
    dictionary for loss computation.

    Attributes:
        _entry_configs: Dictionary mapping dataflow names to their configurations
        _loss_required_names: Set of names required for loss computation
    """

    def __init__(
        self, entry_configs: dict[str, DataEntry], next_step: ProcessingStep | None = None
    ):
        """Initialize the loss dataflow aggregation step.

        Args:
            entry_configs (dict[str, DataEntry]): Mapping from entry names to configurations.
            next_step (ProcessingStep | None): Next step in the processing chain.
        """
        super().__init__(next_step)
        self._entry_configs = entry_configs

        # Pre-compute names required for loss
        self._loss_required_names = {
            name for name, config in entry_configs.items() if config.required_in_loss
        }

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Aggregate loss-required dataflow from all sources.

        Args:
            context (ProcessingContext): Processing context with ``features``, ``targets``,
                ``latents``, and ``predictions`` populated.

        Returns:
            ProcessingContext: Context with ``loss_data`` populated.
        """
        loss_data = {}

        # Collect dataflow from all sources that are required for loss
        all_data = {**context.features, **context.targets, **context.latents, **context.predictions}

        for name, tensor in all_data.items():
            if name in self._loss_required_names and tensor is not None:
                loss_data[name] = tensor

        context.loss_data = loss_data
        return context


class LossPairingStep(ProcessingStep):
    """Pair predictions with targets for supervised loss computation.

    Rules:
    - Strict mapping: for each target name there must be a corresponding prediction key.
    - Single-target fallback: if there is exactly one target and exactly one prediction,
      they are paired even if names differ.
    - Autoencoders: if ``is_autoencoder`` is set and no explicit targets exist, features
      act as targets (feature-wise reconstruction).
    - Fail fast: if required pairs are missing or unexpected prediction keys are present,
      raise a clear error that lists the offending keys and the available targets/predictions.

    The resulting pairs are stored in ``context.loss_data`` as
    ``{target_name: (pred_tensor, target_tensor)}``, which the wrapper consumes verbatim
    to compute the loss.
    """

    def __init__(
        self,
        entry_configs: dict[str, DataEntry],
        *,
        is_autoencoder: bool = False,
        next_step: ProcessingStep | None = None,
    ):
        super().__init__(next_step)
        self._entry_configs = entry_configs
        self._is_autoencoder = is_autoencoder

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Pair predictions with targets and ensure dtype consistency.

        Predictions come from the model and have the model's dtype (set by Lightning).
        Targets come from the dataset and should already have the correct dtype from
        load_array(), but we perform defensive casting to ensure consistency.

        Args:
            context: Processing context with predictions and targets

        Returns:
            Processing context with loss_data populated

        Raises:
            RuntimeError: If prediction-target pairing fails
        """
        # Determine targets: for autoencoder, default to features when targets are empty
        targets = context.targets
        if self._is_autoencoder and not targets and context.features:
            targets = context.features
            context.targets = targets

        preds = context.predictions
        tnames = list(targets.keys())
        pnames = list(preds.keys())

        pairs: dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = {}

        # Single-target fallback: one target and one prediction → pair them
        if len(tnames) == 1 and len(pnames) == 1:
            tname = tnames[0]
            pred = preds[pnames[0]]
            target = targets[tname]

            # Align target dtype with prediction dtype (maintains user's precision)
            if target.is_floating_point() and pred.is_floating_point():
                if target.dtype != pred.dtype:
                    logger.debug(
                        f"Target '{tname}' dtype ({target.dtype}) differs from prediction ({pred.dtype}). "
                        f"Casting target to match prediction for loss computation."
                    )
                    target = target.to(dtype=pred.dtype)

            pairs[tname] = (pred, target)
            context.loss_data = pairs
            return self._next_step.handle(context) if self._next_step else context

        # Strict matching by names
        missing = [t for t in tnames if t not in preds]
        unexpected = [p for p in pnames if p not in targets]
        if missing or unexpected:
            msg_parts = []
            if missing:
                msg_parts.append(f"missing predictions for targets: {missing}")
            if unexpected:
                msg_parts.append(f"unexpected prediction keys: {unexpected}")
            available = f"available targets={tnames}, predictions={pnames}"
            raise RuntimeError(f"Loss pairing failed: {', '.join(msg_parts)}; {available}")

        # Build pairs by matching keys with dtype alignment
        for name in tnames:
            pred = preds[name]
            target = targets[name]

            # Align target dtype with prediction dtype
            if target.is_floating_point() and pred.is_floating_point():
                if target.dtype != pred.dtype:
                    logger.debug(
                        f"Target '{name}' dtype ({target.dtype}) differs from prediction ({pred.dtype}). "
                        f"Casting target to match prediction for loss computation."
                    )
                    target = target.to(dtype=pred.dtype)

            pairs[name] = (pred, target)

        context.loss_data = pairs
        return self._next_step.handle(context) if self._next_step else context


class ValidationDataStep(ProcessingStep):
    """Prepare dataflow for validation/testing (predictions vs targets).

    This step is typically used as a final step in validation pipelines
    to organize predictions and corresponding targets for metric computation.
    """

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Organize validation dataflow for metric computation.

        This step doesn't modify the context but could be extended to
        perform validation-specific dataflow organization.

        Args:
            context (ProcessingContext): Processing context with all

        Returns:
            ProcessingContext: Unchanged context (ready for validation).
        """
        # For now, just pass through - could be extended for validation-specific logic
        return context


class ProcessingPipeline:
    """Complete processing pipeline orchestrating the chain of steps.

    This class provides a high-level interface for creating and executing
    processing pipelines with different configurations for training,
    validation, and inference.

    Attributes:
        _head_step: The first step in the processing chain
    """

    def __init__(self, head_step: ProcessingStep):
        """Initialize the processing pipeline.

        Args:
            head_step: The first step in the processing chain
        """
        self._head_step = head_step

    def execute(self, raw_batch: dict[str, torch.Tensor]) -> ProcessingContext:
        """Execute the complete processing pipeline.

        Args:
            raw_batch: Raw batch dataflow from the dataset

        Returns:
            Complete processing context with all dataflow processed
        """
        context = ProcessingContext()
        context.raw_batch = raw_batch

        return self._head_step.handle(context)

    def execute_with_context(self, context: ProcessingContext) -> ProcessingContext:
        """Execute the pipeline with an existing context.

        Args:
            context: Existing processing context

        Returns:
            Updated processing context
        """
        return self._head_step.handle(context)
