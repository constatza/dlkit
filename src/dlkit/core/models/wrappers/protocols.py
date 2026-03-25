"""SOLID protocols for Lightning wrapper components.

These protocols define the interfaces for the separated concerns in the
ProcessingLightningWrapper architecture:
- ILossComputer: compute scalar loss from predictions + named batch
- IMetricsUpdater: accumulate and expose metric state
- IModelInvoker: extract tensors from TensorDict and call model positionally
- IBatchTransformer: forward-only transform applied every step
- IFittableBatchTransformer: extends IBatchTransformer with fit lifecycle
"""

from typing import Any, Protocol, runtime_checkable

import torch
from tensordict import TensorDict
from torch import Tensor, nn


@runtime_checkable
class ILossComputer(Protocol):
    """Single responsibility: compute scalar loss from predictions + named batch.

    This protocol decouples loss computation from the Lightning wrapper,
    allowing different loss routing strategies without modifying the wrapper.
    """

    def compute(self, predictions: Tensor, batch: Any) -> Tensor:
        """Compute loss from predictions and batch.

        Args:
            predictions: Model output tensor.
            batch: TensorDict containing features and targets.

        Returns:
            Scalar loss tensor.
        """
        ...


@runtime_checkable
class IMetricsUpdater(Protocol):
    """Single responsibility: accumulate and expose metric state.

    This protocol decouples metric routing from the Lightning wrapper.
    Each metric can have its own target key and extra inputs.
    """

    def update(self, predictions: Tensor, batch: Any, stage: str) -> None:
        """Update metrics for a given stage.

        Args:
            predictions: Model output tensor.
            batch: TensorDict containing features and targets.
            stage: Stage identifier ("val" or "test").
        """
        ...

    def compute(self, stage: str) -> dict[str, Any]:
        """Compute accumulated metric values.

        Args:
            stage: Stage identifier ("val" or "test").

        Returns:
            Dictionary mapping metric names to computed values.
        """
        ...

    def reset(self, stage: str) -> None:
        """Reset metric state.

        Args:
            stage: Stage identifier ("val" or "test").
        """
        ...


@runtime_checkable
class IModelInvoker(Protocol):
    """Single responsibility: invoke model and return enriched TensorDict.

    The model invoker reads feature tensors from *batch*, calls the model,
    and writes the output back into the batch under a ``"predictions"`` key
    (and optionally latent keys).  Callers read ``batch["predictions"]``
    after calling ``invoke()``.
    """

    def invoke(self, model: nn.Module, batch: TensorDict) -> TensorDict:
        """Invoke model and return batch enriched with ``"predictions"``.

        Args:
            model: PyTorch model to invoke.
            batch: TensorDict containing features and targets.

        Returns:
            Enriched TensorDict with ``"predictions"`` key added (and
            optionally latent keys such as ``("latents", "mu")``).
        """
        ...


@runtime_checkable
class IBatchTransformer(Protocol):
    """Forward-only transform: called every training, validation, and prediction step.

    This protocol separates transform application from the Lightning wrapper.
    """

    def transform(self, batch: Any) -> Any:
        """Apply transforms to the batch.

        Args:
            batch: Input TensorDict.

        Returns:
            Transformed TensorDict.
        """
        ...

    def inverse_transform_predictions(
        self, predictions: Tensor | TensorDict, target_key: str
    ) -> Tensor | TensorDict:
        """Apply inverse target transform to predictions.

        Used in predict_step to convert predictions back to original space.
        Single-head (Tensor): uses target_key to look up chain.
        Multi-head (TensorDict): applies per-key chain lookup; target_key ignored.

        Args:
            predictions: Model output in transformed space — Tensor or TensorDict.
            target_key: Name of the target entry whose chain to invert (single-head only).

        Returns:
            Predictions in original (untransformed) space, same type as input.
        """
        ...


@runtime_checkable
class IFittableBatchTransformer(IBatchTransformer, Protocol):
    """Extends IBatchTransformer with fit lifecycle.

    Used only during on_fit_start. Transforms that require fitting
    (e.g. MinMaxScaler, StandardScaler, PCA) should be wrapped in
    an IFittableBatchTransformer.
    """

    def fit(self, dataloader: Any) -> None:
        """Fit all fittable transforms using training data.

        Args:
            dataloader: Training DataLoader to iterate for fitting.
        """
        ...

    def is_fitted(self) -> bool:
        """Check if all fittable transforms are fitted.

        Returns:
            True if all transforms are fitted or there are no fittable transforms.
        """
        ...


@runtime_checkable
class IPredictionStrategy(Protocol):
    """Single responsibility: encapsulate predict_step logic.

    Strategies replace the hardcoded ``predict_step`` in the base wrapper,
    allowing discriminative and generative inference to coexist without
    modifying the training loop.

    Standard models inject ``DiscriminativePredictionStrategy``.
    Generative models inject ``ODEPredictionStrategy`` (or equivalent).
    """

    def predict(
        self,
        model: nn.Module,
        batch: Any,
        generator: torch.Generator | None,
    ) -> TensorDict:
        """Run inference and return a structured output TensorDict.

        Args:
            model: PyTorch model to invoke.
            batch: Input TensorDict from the dataloader.
            generator: Optional RNG (for reproducible generation).

        Returns:
            TensorDict with at minimum ``"predictions"``, ``"targets"``,
            and ``"latents"`` keys.
        """
        ...


@runtime_checkable
class IConfigurablePredictionStrategy(IPredictionStrategy, Protocol):
    """IPredictionStrategy that also exposes shape configuration for ODE-based strategies."""

    def configure_shape(self, data_shape: tuple[int, ...]) -> None:
        """Set the per-sample data shape for noise generation.

        Args:
            data_shape: Spatial dimensions (excluding batch), e.g. ``(3, 32, 32)``.
        """
        ...

    @property
    def data_shape(self) -> tuple[int, ...] | None:
        """Per-sample data shape, or None if not yet configured.

        Returns:
            Shape tuple or None.
        """
        ...

    @property
    def n_steps(self) -> int:
        """Number of ODE integration steps.

        Returns:
            Integer step count.
        """
        ...


@runtime_checkable
class IBatchTransform(Protocol):
    """Single-call coupled supervision transform applied per batch.

    Unlike ``IBatchTransformer`` (which applies per-slot normalisation chains),
    ``IBatchTransform`` implements *coupled* multi-tensor operations that span
    features and targets together — e.g. flow matching supervision builders
    that sample noise, time, and compute velocity targets in one call.

    The generator argument enables reproducible stochastic transforms.
    """

    def __call__(
        self,
        batch: TensorDict,
        generator: torch.Generator | None,
    ) -> TensorDict:
        """Apply the coupled transform to a batch.

        Args:
            batch: Input TensorDict.
            generator: Optional RNG for reproducibility.

        Returns:
            Transformed TensorDict (may be mutated in place or a new object).
        """
        ...


@runtime_checkable
class IGeneratorFactory(Protocol):
    """Single responsibility: produce (or withhold) a ``torch.Generator`` per batch.

    Injected into the wrapper so training and validation reproducibility
    strategies are swappable without modifying the training loop.
    """

    def __call__(self, batch_idx: int) -> torch.Generator | None:
        """Return a generator for this batch, or None to use global RNG.

        Args:
            batch_idx: Current batch index.

        Returns:
            A seeded ``torch.Generator``, or ``None``.
        """
        ...
