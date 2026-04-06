"""ContinuousFlowWrapper — Level 2 generative wrapper.

Fixes inference: ``predict_step`` always uses ODE integration.
Captures per-sample data shape from the first training batch via ``on_fit_start``.
Saves ODE metadata to checkpoints.

Subclasses specialize training supervision and loss only (Level 3).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tensordict import TensorDict

from dlkit.engine.adapters.lightning.generative import GenerativeLightningWrapper

if TYPE_CHECKING:
    from dlkit.engine.adapters.lightning.protocols import IConfigurablePredictionStrategy


class ContinuousFlowWrapper(GenerativeLightningWrapper):
    """Base for all continuous-time flow models.

    Fixes: ``predict_step`` uses ``ODEPredictionStrategy`` with the model as dynamics.
    Captures data shape from the first training batch for noise generation.
    Subclasses specialise training supervision + loss only.

    Args:
        ode_prediction_strategy: ODE-based prediction strategy satisfying
            ``IConfigurablePredictionStrategy``.  Shape is configured automatically
            via ``on_fit_start``.
        **kwargs: Forwarded to ``GenerativeLightningWrapper.__init__``.
    """

    def __init__(
        self,
        *,
        ode_prediction_strategy: IConfigurablePredictionStrategy,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs, prediction_strategy=ode_prediction_strategy)
        self._ode_strategy = ode_prediction_strategy

    def on_fit_start(self) -> None:
        """Configure ODE strategy shape from first training batch.

        Iterates the first batch of the training dataloader, infers the
        per-sample feature shape, and calls
        ``_ode_strategy.configure_shape()``.
        Also triggers batch transformer fitting (base class behaviour).
        """
        super().on_fit_start()
        trainer = getattr(self, "trainer", None)
        if trainer is None or not hasattr(trainer, "datamodule"):
            return
        dm = trainer.datamodule
        if dm is None or not hasattr(dm, "train_dataloader"):
            return
        loader = dm.train_dataloader()
        for batch in loader:
            shape = self._infer_data_shape(batch)
            self._ode_strategy.configure_shape(shape)
            break

    def _infer_data_shape(self, batch: TensorDict) -> tuple[int, ...]:
        """Extract per-sample shape from the first feature tensor.

        Override in subclasses that use a non-standard features layout.

        Args:
            batch: A TensorDict batch from the training dataloader.

        Returns:
            Per-sample shape tuple (excluding batch dimension).
        """
        features = batch.get("features", batch)
        try:
            first = next(iter(features.values()))
            return tuple(first.shape[1:])
        except StopIteration:
            return ()

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Augment checkpoint with continuous-flow ODE metadata.

        Args:
            checkpoint: Checkpoint dict to augment.
        """
        super().on_save_checkpoint(checkpoint)
        checkpoint.setdefault("dlkit_metadata", {})["continuous_flow"] = {
            "data_shape": list(self._ode_strategy.data_shape or []),
            "n_steps": self._ode_strategy.n_steps,
        }

    def forward(self, x: Any, t: Any) -> Any:
        """Forward pass: delegate to underlying model with (x, t) signature.

        Args:
            x: Input tensor or TensorDict.
            t: Time tensor.

        Returns:
            Model output (velocity field).
        """
        return self.model(x, t)
