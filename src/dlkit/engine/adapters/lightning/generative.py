"""GenerativeLightningWrapper — base for all generative models.

This class implements the standard training loop for generative models:
coupled supervision transforms + loss computation. It's a marker base for
factory routing while providing concrete _run_step implementation.

It signals "this is a generative model" for:
- ``isinstance`` checks in factory routing
- ``issubclass`` guards in ``WrapperFactory`` / ``BuildFactory``
- Documentation and type annotations
"""

from __future__ import annotations

from typing import Any, cast

from torch import Tensor

from dlkit.engine.adapters.lightning.base import ProcessingLightningWrapper


class GenerativeLightningWrapper(ProcessingLightningWrapper):
    """Base for any generative Lightning wrapper with standard training loop.

    Guarantees:
    - ``batch_transforms`` is non-empty (coupled supervision injected).
    - ``prediction_strategy`` is a generative strategy (e.g. ODE-based).
    - Implements _run_step with the standard training loop.

    Subclasses specialise supervision signal and loss (Level 3) or inference
    strategy (Level 2) without changing the training loop in the base class.

    Use ``isinstance(wrapper, GenerativeLightningWrapper)`` to branch on
    generative vs. discriminative behaviour in factory code.

    Transform fitting is not handled here — it runs once, deterministically,
    during the build phase (``engine.training.transform_fitting``), before any
    Trainer exists.
    """

    def _run_step(self, batch: Any, batch_idx: int, stage: str) -> tuple[Tensor, int | None, Any]:
        """Execute one forward+loss step for generative models.

        Applies batch transforms (coupled supervision), applies per-slot transforms,
        invokes the model, and computes loss. This matches the standard training
        loop from StandardLightningWrapper.

        Args:
            batch: Input batch from dataset.
            batch_idx: Index of the batch.
            stage: Stage identifier ('train', 'val', 'test').

        Returns:
            Tuple of (loss, batch_size, enriched_batch).
        """
        from functools import reduce

        from dlkit.engine.adapters.lightning.base import _batch_size_of

        gen = (self._train_generator_factory if stage == "train" else self._val_generator_factory)(
            batch_idx
        )

        # Apply coupled supervision transforms (specific to generative models)
        if self._batch_transforms:
            batch = reduce(lambda b, t: t(b, gen), self._batch_transforms, batch)

        # Apply per-slot normalisation chains
        batch = self._batch_transformer.transform(batch)

        # Invoke the model
        batch = self._model_invoker.invoke(self.model, batch)

        # Compute loss
        loss = self._loss_computer.compute(cast(Tensor, batch["predictions"]), batch)

        batch_size = _batch_size_of(batch["predictions"])
        return loss, batch_size, batch
