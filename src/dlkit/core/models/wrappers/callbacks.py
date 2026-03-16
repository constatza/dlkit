"""Lightning callbacks for wrapper lifecycle concerns.

Provides reusable Lightning Callbacks that replace lifecycle methods
previously embedded in the wrapper classes:
- TransformFittingCallback: Fits NamedBatchTransformer before training starts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lightning import Callback
from loguru import logger

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer


class TransformFittingCallback(Callback):
    """Fits a batch transformer on the training dataloader before training starts.

    Replaces ``StandardLightningWrapper.on_fit_start()`` with a proper
    Lightning Callback, separating the transform-fitting concern from the
    training loop wrapper.

    The callback is a no-op when the transformer is already fitted or does
    not implement ``IFittableBatchTransformer``.

    Args:
        batch_transformer: The batch transformer to fit. Fitting is skipped
            unless it implements ``IFittableBatchTransformer``.

    Example:
        ```python
        callback = TransformFittingCallback(batch_transformer)
        trainer = Trainer(callbacks=[callback])
        ```
    """

    def __init__(self, batch_transformer: Any) -> None:
        """Initialize with the batch transformer to manage.

        Args:
            batch_transformer: Transformer to fit; may or may not be fittable.
        """
        super().__init__()
        self._batch_transformer = batch_transformer

    def on_fit_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        """Fit the batch transformer if it is fittable and not yet fitted.

        Called automatically by Lightning before the first training epoch.

        Args:
            trainer: The Lightning Trainer driving the fit.
            pl_module: The LightningModule being trained (unused).
        """
        from dlkit.core.models.wrappers.protocols import IFittableBatchTransformer

        if not isinstance(self._batch_transformer, IFittableBatchTransformer):
            return
        if self._batch_transformer.is_fitted():
            return
        dm = getattr(trainer, "datamodule", None)
        if dm is None or not hasattr(dm, "train_dataloader"):
            return
        loader = dm.train_dataloader()
        logger.info("Starting transform fitting from training dataloader.")
        self._batch_transformer.fit(loader)
        logger.info("Finished transform fitting.")
