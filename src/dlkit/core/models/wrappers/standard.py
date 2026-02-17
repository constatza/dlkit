"""Standard Lightning wrapper for tensor-based models with positional transform support.

This module provides a Lightning wrapper that extends the base wrapper with
transform capabilities using positional ModuleList chains rather than string-keyed
ModuleDicts, eliminating string dispatch throughout the hot path.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from torch.nn import ModuleList, Identity

from dlkit.tools.config import ModelComponentSettings, WrapperComponentSettings
from dlkit.tools.config.data_entries import DataEntry, is_feature_entry, is_target_entry
from dlkit.core.training.transforms.chain import TransformChain
from dlkit.core.training.transforms.base import FittableTransform, InvertibleTransform
from dlkit.core.datatypes.batch import Batch
from .base import ProcessingLightningWrapper
from .functions import apply_chain, apply_inverse_chain

if TYPE_CHECKING:
    from dlkit.core.shape_specs.simple_inference import ShapeSummary


def _navigate_and_register(module: torch.nn.Module, key: str, value: Tensor) -> None:
    """Navigate a module hierarchy and register a missing buffer at the leaf node.

    Transforms like MinMaxScaler allocate ``min``/``max`` buffers lazily during
    ``fit()``. This helper pre-registers them so ``load_state_dict`` can fill
    the values correctly.

    Args:
        module: Root module to start navigation from.
        key: Dot-separated path to the buffer (e.g. ``"transforms.0.min"``).
        value: Tensor to register as buffer.
    """
    parts = key.split(".")
    current = module
    for part in parts[:-1]:
        if part.isdigit():
            idx = int(part)
            if isinstance(current, ModuleList):
                current = current[idx]
            elif hasattr(current, "transforms"):
                current = current.transforms[idx]  # type: ignore[index]
            else:
                return
        else:
            if not hasattr(current, part):
                return
            current = getattr(current, part)
    buffer_name = parts[-1]
    if not hasattr(current, buffer_name):
        current.register_buffer(buffer_name, value)


class StandardLightningWrapper(ProcessingLightningWrapper):
    """Lightning wrapper for standard tensor-based neural networks with positional transforms.

    Transform chains are stored as positional ModuleLists aligned with entry_configs
    insertion order. Index 0 = first Feature entry, etc. No string dispatch anywhere.

    Attributes:
        _feature_chains (torch.nn.ModuleList): Positional feature transform chains.
        _target_chains (torch.nn.ModuleList): Positional target transform chains.

    Example:
        ```python
        wrapper = StandardLightningWrapper(
            settings=wrapper_settings,
            model_settings=model_settings,
            shape_spec=shape_spec,
            entry_configs=data_configs,
        )
        ```
    """

    def __init__(
        self,
        *,
        settings: WrapperComponentSettings,
        model_settings: ModelComponentSettings,
        entry_configs: tuple[DataEntry, ...] | None = None,
        shape_summary: "ShapeSummary | None" = None,
        **kwargs,
    ):
        """Initialize the standard Lightning wrapper.

        Pre-allocates positional transform chains from entry_configs so Lightning's
        load_state_dict can restore fitted state directly without manual reconstruction.

        Args:
            settings: Wrapper configuration settings.
            model_settings: Model configuration settings.
            entry_configs: Data entry configurations (positional, insertion order = position).
            shape_summary: Shape summary from dataset inference (preferred).
            **kwargs: Additional arguments passed to base class.
        """
        super().__init__(
            settings=settings,
            model_settings=model_settings,
            entry_configs=entry_configs,
            shape_summary=shape_summary,
            **kwargs,
        )

        feature_entries = [e for e in self._entry_configs if is_feature_entry(e)]
        target_entries = [e for e in self._entry_configs if is_target_entry(e)]

        self._feature_chains: ModuleList = ModuleList(
            self._make_chain(e) for e in feature_entries
        )
        self._target_chains: ModuleList = ModuleList(
            self._make_chain(e) for e in target_entries
        )

    def _make_chain(self, entry: DataEntry) -> torch.nn.Module:
        """Create a TransformChain for an entry, or Identity if no transforms configured.

        Args:
            entry: Data entry configuration with optional transforms attribute.

        Returns:
            TransformChain if transforms configured, nn.Identity otherwise.
        """
        settings = getattr(entry, "transforms", None)
        return TransformChain(settings) if settings else Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the model with tensor input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Model output tensor.
        """
        return self.model(x)

    # =============================================================================
    # Positional Transform Helpers
    # =============================================================================

    def _apply_feature_transforms(self, batch: Batch) -> Batch:
        """Apply feature chains positionally. Returns new Batch with transformed features.

        Args:
            batch: Input batch with raw feature tensors.

        Returns:
            New Batch with transformed features.
        """
        if not self._feature_chains:
            return batch
        transformed = tuple(
            apply_chain(x, c) for x, c in zip(batch.features, self._feature_chains)
        )
        return replace(batch, features=transformed)

    def _apply_target_transforms(self, batch: Batch) -> Batch:
        """Apply target chains positionally. Returns new Batch with transformed targets.

        Args:
            batch: Input batch with raw target tensors.

        Returns:
            New Batch with transformed targets.
        """
        if not self._target_chains:
            return batch
        transformed = tuple(
            apply_chain(x, c) for x, c in zip(batch.targets, self._target_chains)
        )
        return replace(batch, targets=transformed)

    def _apply_inverse_target_transforms(self, batch: Batch) -> Batch:
        """Apply inverse target chains positionally. Returns new Batch in original space.

        Args:
            batch: Input batch with normalized target tensors.

        Returns:
            New Batch with targets back in original space.
        """
        if not self._target_chains:
            return batch
        transformed = tuple(
            apply_inverse_chain(x, c) for x, c in zip(batch.targets, self._target_chains)
        )
        return replace(batch, targets=transformed)

    def _chains_are_fitted(self) -> bool:
        """True if all non-Identity chains already have fitted state (from checkpoint).

        Returns:
            True when all FittableTransform chains are fitted, False otherwise.
        """
        for chain in [*self._feature_chains, *self._target_chains]:
            if isinstance(chain, FittableTransform) and not chain.fitted:
                return False
        return True

    # =============================================================================
    # Overridden Step Methods (Positional Batch-First)
    # =============================================================================

    def training_step(self, batch: Batch, batch_idx: int) -> dict[str, Any]:
        """Training step with positional transform application.

        Args:
            batch (Batch): Positional batch from dataset.
            batch_idx (int): Index of the batch.

        Returns:
            dict[str, Any]: Dictionary containing the training loss.
        """
        batch = self._apply_feature_transforms(batch)
        batch = self._apply_target_transforms(batch)
        predictions = self._invoke_model(batch)
        loss = self._compute_loss(predictions, batch.targets)
        self._log_stage_outputs("train", loss)
        return {"loss": loss}

    def validation_step(self, batch: Batch, batch_idx: int) -> dict[str, Any]:
        """Validation step with positional transform application.

        Args:
            batch (Batch): Positional batch from dataset.
            batch_idx (int): Index of the batch.

        Returns:
            dict[str, Any]: Dictionary containing validation metrics.
        """
        batch = self._apply_feature_transforms(batch)
        batch = self._apply_target_transforms(batch)
        predictions = self._invoke_model(batch)
        val_loss = self._compute_loss(predictions, batch.targets)
        metrics = self._update_metrics(predictions, batch.targets, stage="val")
        self._log_stage_outputs("val", val_loss, metrics)
        return {"val_loss": val_loss}

    def test_step(self, batch: Batch, batch_idx: int) -> dict[str, Any]:
        """Test step with positional transform application.

        Args:
            batch (Batch): Positional batch from dataset.
            batch_idx (int): Index of the batch.

        Returns:
            dict[str, Any]: Dictionary containing test metrics.
        """
        batch = self._apply_feature_transforms(batch)
        batch = self._apply_target_transforms(batch)
        predictions = self._invoke_model(batch)
        test_loss = self._compute_loss(predictions, batch.targets)
        metrics = self._update_metrics(predictions, batch.targets, stage="test")
        self._log_stage_outputs("test", test_loss, metrics)
        return {"test_loss": test_loss}

    def predict_step(self, batch: Batch, batch_idx: int) -> dict[str, Any]:
        """Prediction step with positional transform application and inverse.

        Applies forward transforms to features, runs model, then applies inverse
        target transform to predictions so outputs are in original data space.

        Args:
            batch (Batch): Positional batch from dataset.
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary with ``predictions``, ``targets``, and ``latents`` as tuples.
        """
        original_targets = batch.targets
        batch = self._apply_feature_transforms(batch)
        predictions = self._invoke_model(batch)

        # Apply inverse target transform to predictions (model output → original space)
        if isinstance(predictions, Tensor) and self._target_chains:
            predictions = apply_inverse_chain(predictions, self._target_chains[0])

        return {
            "predictions": (predictions,),
            "targets": original_targets,
            "latents": (),
        }

    # =============================================================================
    # Transform Fitting (on_fit_start)
    # =============================================================================

    def on_fit_start(self) -> None:
        """Fit all entry-based transform chains using the entire training dataloader.

        Guards: skips if chains are already fitted (loaded from checkpoint),
        no trainer/datamodule is available, or no transforms are configured.
        Aggregates full training data per entry position and fits once.
        """
        if self._chains_are_fitted():
            return

        trainer = getattr(self, "trainer", None)
        if trainer is None or not hasattr(trainer, "datamodule"):
            return
        dm = trainer.datamodule
        if dm is None or not hasattr(dm, "train_dataloader"):
            return

        feature_entries = [e for e in self._entry_configs if is_feature_entry(e)]
        target_entries = [e for e in self._entry_configs if is_target_entry(e)]

        if not any(getattr(e, "transforms", None) for e in feature_entries + target_entries):
            return

        try:
            loader = dm.train_dataloader()
        except Exception:
            return

        feat_buffers: list[list[Tensor]] = [[] for _ in feature_entries]
        tgt_buffers: list[list[Tensor]] = [[] for _ in target_entries]

        for batch in loader:
            for i, t in enumerate(batch.features):
                feat_buffers[i].append(t)
            for i, t in enumerate(batch.targets):
                tgt_buffers[i].append(t)

        for chain, entry, bufs in zip(self._feature_chains, feature_entries, feat_buffers):
            if bufs and getattr(entry, "transforms", None) and isinstance(chain, FittableTransform):
                chain.fit(torch.cat(bufs, dim=0))

        for chain, entry, bufs in zip(self._target_chains, target_entries, tgt_buffers):
            if bufs and getattr(entry, "transforms", None) and isinstance(chain, FittableTransform):
                chain.fit(torch.cat(bufs, dim=0))

    # =============================================================================
    # Checkpoint Hooks
    # =============================================================================

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Save inference metadata to checkpoint.

        The _feature_chains and _target_chains ModuleLists are persisted
        automatically by Lightning's state_dict mechanism.

        Args:
            checkpoint: Checkpoint dict to augment.
        """
        super().on_save_checkpoint(checkpoint)

        feature_names = [e.name for e in self._entry_configs if is_feature_entry(e)]
        target_names = [e.name for e in self._entry_configs if is_target_entry(e)]

        checkpoint["inference_metadata"] = {
            "entry_configs": self._entry_configs,
            "wrapper_settings": (
                self._wrapper_settings.model_dump()
                if hasattr(self._wrapper_settings, "model_dump")
                else dict(self._wrapper_settings)
            ),
            "feature_names": feature_names,
            "target_names": target_names,
            "model_shape": getattr(self, "shape", None),
        }

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True, assign: bool = False) -> Any:
        """Override to pre-register lazy-allocated buffers before the standard load.

        Transforms like MinMaxScaler register ``min``/``max`` buffers lazily during
        ``fit()``. PyTorch's ``load_state_dict`` silently skips keys that aren't
        already registered (when ``strict=False``). This override pre-registers any
        missing buffers so the standard load can fill their values.

        Args:
            state_dict: State dictionary to load.
            strict: Whether to strictly enforce key matching.
            assign: Whether to use assign semantics.

        Returns:
            Result of the standard load_state_dict call.
        """
        self._pre_register_lazy_buffers(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _pre_register_lazy_buffers(self, state_dict: dict[str, Any]) -> None:
        """Pre-register lazy-allocated buffers found in state_dict.

        Iterates positional chains and for each one extracts its sub-state-dict,
        calls load_state_dict to see which keys are unexpected (i.e. not yet
        registered), then navigates the module hierarchy to register them.

        Args:
            state_dict: Full model state dictionary.
        """
        pairs = [("_feature_chains", self._feature_chains), ("_target_chains", self._target_chains)]
        for prefix, chains in pairs:
            for idx, chain in enumerate(chains):
                chain_prefix = f"{prefix}.{idx}."
                chain_state = {
                    k[len(chain_prefix):]: v
                    for k, v in state_dict.items()
                    if k.startswith(chain_prefix)
                }
                if not chain_state:
                    continue
                result = chain.load_state_dict(chain_state, strict=False)
                for key in result.unexpected_keys:
                    if key in chain_state:
                        _navigate_and_register(chain, key, chain_state[key])

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore checkpoint. Chains are pre-allocated in __init__ so load_state_dict
        restores their fitted state automatically.

        Args:
            checkpoint: Checkpoint dict to restore from.
        """
        super().on_load_checkpoint(checkpoint)


class BareWrapper(StandardLightningWrapper):
    """Minimal Lightning wrapper with basic functionality.

    This is a simplified version of StandardLightningWrapper that provides
    minimal functionality without the full transform integration.
    Useful for simple models that don't need complex data processing.
    """

    def __init__(self, model_settings: ModelComponentSettings, **kwargs):
        """Initialize the bare wrapper.

        Args:
            model_settings: Model configuration settings
            **kwargs: Additional arguments (most will be ignored)
        """
        from dlkit.tools.config import WrapperComponentSettings

        minimal_settings = WrapperComponentSettings()

        super().__init__(
            settings=minimal_settings,
            model_settings=model_settings,
            **kwargs
        )

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Any]:
        """Simplified training step without transform processing.

        Args:
            batch: Raw batch data
            batch_idx: Index of the batch

        Returns:
            Dictionary containing the training loss
        """
        x = next(iter(batch.values()))
        output = self.forward(x)

        batch_values = list(batch.values())
        if len(batch_values) >= 2:
            target = batch_values[1]
            loss = self.loss_function(output, target)
        else:
            loss = torch.tensor(0.0, requires_grad=True)

        self._log_stage_outputs("train", loss)
        return {"loss": loss}

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Any]:
        """Simplified validation step without transform processing.

        Args:
            batch: Raw batch data
            batch_idx: Index of the batch

        Returns:
            Dictionary containing validation metrics
        """
        x = next(iter(batch.values()))
        output = self.forward(x)

        batch_values = list(batch.values())
        metrics = None
        if len(batch_values) >= 2:
            target = batch_values[1]
            val_loss = self.loss_function(output, target)

            if self.val_metrics:
                metrics = self.val_metrics(output, target)
        else:
            val_loss = torch.tensor(0.0)

        self._log_stage_outputs("val", val_loss, metrics)
        return {"val_loss": val_loss}

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Any]:
        """Simplified test step without transform processing.

        Args:
            batch: Raw batch data
            batch_idx: Index of the batch

        Returns:
            Dictionary containing test metrics
        """
        x = next(iter(batch.values()))
        output = self.forward(x)

        batch_values = list(batch.values())
        metrics = None
        if len(batch_values) >= 2:
            target = batch_values[1]
            test_loss = self.loss_function(output, target)

            if self.test_metrics:
                metrics = self.test_metrics(output, target)
        else:
            test_loss = torch.tensor(0.0)

        self._log_stage_outputs("test", test_loss, metrics)
        return {"test_loss": test_loss}
