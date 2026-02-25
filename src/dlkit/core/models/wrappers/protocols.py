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

import torch.nn as nn
from torch import Tensor


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
    """Single responsibility: extract tensors from TensorDict and call model positionally.

    This protocol decouples model invocation from the Lightning wrapper.
    The standard implementation extracts model_input features in config order.
    """

    def invoke(self, model: nn.Module, batch: Any) -> Tensor:
        """Invoke model with tensors extracted from batch.

        Args:
            model: PyTorch model to invoke.
            batch: TensorDict containing features and targets.

        Returns:
            Model output tensor.
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

    def inverse_transform_predictions(self, predictions: Tensor, target_key: str) -> Tensor:
        """Apply inverse target transform to predictions.

        Used in predict_step to convert predictions back to original space.

        Args:
            predictions: Model output tensor in transformed space.
            target_key: Name of the target entry whose chain to invert.

        Returns:
            Predictions in original (untransformed) space.
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
