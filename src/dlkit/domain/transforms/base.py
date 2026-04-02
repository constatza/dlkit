"""Base class for tensor transformations with simplified design.

This module provides the foundation Transform class that all transforms inherit from.
Following Python's duck typing philosophy, most capabilities are optional methods.

Design Philosophy:
- Most capabilities optional via method overrides (fit, configure_shape)
- One Protocol for invertible transforms (type safety + early validation)
- Duck typing for fittable/shape-aware capabilities
- Single source of truth: nn.Module for device management and checkpointing
- Fitted state stored as tensor buffer for checkpoint persistence

Example:
    >>> class MyTransform(Transform):
    ...     def forward(self, x: torch.Tensor) -> torch.Tensor:
    ...         return x * 2
    ...
    ...     def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
    ...         return x / 2
"""

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
from torch import nn

if TYPE_CHECKING:
    from dlkit.shared.shapes import ShapeSpecProtocol


@runtime_checkable
class FittableTransform(Protocol):
    """Protocol for transforms that require fitting to data.

    Transforms implementing this Protocol must provide a fit() method that learns
    statistics from training data before the transform can be used.

    Example:
        >>> class StandardScaler(Transform):
        ...     def fit(self, data: torch.Tensor) -> None:
        ...         self.mean = data.mean(dim=0)
        ...         self.std = data.std(dim=0)
        ...         self.fitted = True
        >>>
        >>> scaler = StandardScaler()
        >>> isinstance(scaler, FittableTransform)  # True
    """

    def fit(self, data: torch.Tensor) -> None:
        """Fit transform parameters to training data.

        Args:
            data: Training data tensor to compute statistics from.
        """
        ...


@runtime_checkable
class IncrementalFittableTransform(Protocol):
    """Protocol for transforms that can be fitted incrementally on data batches.

    Incremental transforms support streaming training-data passes without
    materializing the full dataset in memory.
    """

    def reset_fit_state(self) -> None:
        """Reset internal fitting accumulators before a new fit pass."""
        ...

    def update_fit(self, batch: torch.Tensor) -> None:
        """Update fitting accumulators from one data batch."""
        ...

    def finalize_fit(self) -> None:
        """Finalize accumulators into fitted buffers and mark transform fitted."""
        ...


@runtime_checkable
class InvertibleTransform(Protocol):
    """Protocol for transforms that support inverse transformation.

    Use this Protocol for type safety in transform chains. The Protocol enables:
    - Type checking at chain creation time
    - Clear documentation of invertibility requirement
    - Early validation instead of runtime errors

    Example:
        >>> class MinMaxScaler(Transform):
        ...     def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...         return 2 * (x - self.min) / (self.max - self.min) - 1
        ...
        ...     def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        ...         return (x + 1) / 2 * (self.max - self.min) + self.min
        >>>
        >>> scaler = MinMaxScaler(dim=0)
        >>> isinstance(scaler, InvertibleTransform)  # True
    """

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse the transformation.

        Args:
            x: Transformed tensor to invert.

        Returns:
            Tensor in original space.
        """
        ...


@runtime_checkable
class ShapeAwareTransform(Protocol):
    """Protocol for transforms that benefit from shape information.

    Transforms implementing this Protocol can receive shape information via
    configure_shape() for eager buffer allocation (performance optimization).

    Example:
        >>> class MinMaxScaler(Transform):
        ...     def configure_shape(self, shape_spec, entry_name: str) -> None:
        ...         shape = shape_spec.get_shape(entry_name)
        ...         moments_shape = self._compute_moments_shape(shape)
        ...         self.register_buffer("min", torch.zeros(moments_shape))
        ...         self.register_buffer("max", torch.ones(moments_shape))
        >>>
        >>> scaler = MinMaxScaler(dim=0)
        >>> isinstance(scaler, ShapeAwareTransform)  # True
    """

    def configure_shape(self, shape_spec: ShapeSpecProtocol, entry_name: str) -> None:
        """Configure transform with shape information.

        Args:
            shape_spec: Shape specification containing entry shapes.
            entry_name: Name of the entry to get shape for.
        """
        ...


class Transform(nn.Module):
    """Base class for tensor transformations.

    This class provides the foundation for all transforms in DLKit. It integrates
    with PyTorch's nn.Module for device management and checkpoint persistence.

    **Simplified Design (Phase 2 Refactoring):**
    - All methods except forward() are OPTIONAL
    - No ABC mixins required - override methods as needed
    - Capability checking via duck typing (hasattr/try-except)
    - More Pythonic, less ceremony

    **Capabilities (all optional):**
    - **Fittable**: Override fit() to learn statistics from data
    - **Invertible**: Override inverse_transform() to reverse the transformation
    - **Shape-aware**: Override configure_shape() for eager buffer allocation
    - **Serializable**: Automatic via nn.Module.state_dict() (no custom method needed)

    **Shape Handling:**
    - Transforms don't store input_shape (duplicates shape_spec system)
    - Shape-aware transforms receive shapes via configure_shape()
    - Shape-agnostic transforms work with any compatible tensor
    - Lazy allocation during fit() is always supported as fallback

    Example:
        >>> # Simple invertible transform
        >>> class DoubleTransform(Transform):
        ...     def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...         return x * 2
        ...
        ...     def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        ...         return x / 2
        >>>
        >>> # Fittable + invertible + shape-aware transform
        >>> class MyScaler(Transform):
        ...     def configure_shape(self, shape_spec, entry_name):
        ...         # Pre-allocate buffers
        ...         shape = shape_spec.get_shape(entry_name)
        ...         self.register_buffer("mean", torch.zeros(shape))
        ...
        ...     def fit(self, data: torch.Tensor) -> None:
        ...         self.mean = data.mean(dim=0)
        ...         self.fitted = True
        ...
        ...     def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...         return x - self.mean
        ...
        ...     def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        ...         return x + self.mean
    """

    apply_inverse: bool
    _fitted: torch.Tensor

    def __init__(self) -> None:
        """Initialize the transform.

        Note:
            Shape information is no longer passed to __init__(). Shape-aware transforms
            should override configure_shape() and receive shapes from the shape_spec system,
            or allocate buffers lazily during fit() from data.shape.
        """
        super().__init__()
        self.apply_inverse = True
        self.register_buffer("_fitted", torch.zeros(1, requires_grad=False))

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the transformation to the input tensor.

        This is the ONLY required method. All others are optional.

        Args:
            x: Input tensor to transform.

        Returns:
            Transformed tensor.

        Example:
            >>> class MyTransform(Transform):
            ...     def forward(self, x):
            ...         return x * 2
        """
        ...

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward() - applies the transformation.

        Args:
            x: Input tensor to transform.

        Returns:
            Transformed tensor.
        """
        return self.forward(x)

    # Optional capabilities - override as needed

    def fit(self, data: torch.Tensor) -> None:
        """Fit transform parameters to training data (OPTIONAL).

        Override this method if your transform needs to learn statistics from data
        (e.g., mean/std, min/max, PCA components). After fitting, set self.fitted = True.

        Default implementation does nothing (for transforms that don't need fitting).

        Args:
            data: Training data tensor to compute statistics from.

        Example:
            >>> class StandardScaler(Transform):
            ...     def fit(self, data):
            ...         self.mean = data.mean(dim=0)
            ...         self.std = data.std(dim=0)
            ...         self.fitted = True
        """

    # NOTE: No default inverse_transform() method!
    # Only transforms that actually implement it should be considered InvertibleTransform.
    # This prevents false positives from Protocol isinstance() checks.
    #
    # If a transform is invertible, it should:
    # 1. Define inverse_transform(self, x: torch.Tensor) -> torch.Tensor
    # 2. The Protocol will automatically recognize it via structural typing
    #
    # Example:
    #     class MinMaxScaler(Transform):  # Will pass isinstance(InvertibleTransform)
    #         def inverse_transform(self, x):
    #             return (x + 1) / 2 * (self.max - self.min) + self.min
    #
    #     class Permutation(Transform):  # Will also pass (has inverse_transform)
    #         def inverse_transform(self, x):
    #             return x.permute(self._inverse_dims)
    #
    #     class NonInvertible(Transform):  # Will NOT pass (no inverse_transform method)
    #         pass

    def configure_shape(self, shape_spec: ShapeSpecProtocol, entry_name: str) -> None:
        """Configure transform with shape information (OPTIONAL).

        Override this method if your transform benefits from eager buffer allocation.
        This is a PERFORMANCE optimization, not a requirement.

        Default implementation does nothing (transforms can allocate lazily during fit()).

        Args:
            shape_spec: Shape specification containing all entry shapes.
            entry_name: Name of the entry to get shape for.

        Example:
            >>> class MinMaxScaler(Transform):
            ...     def configure_shape(self, shape_spec, entry_name):
            ...         shape = shape_spec.get_shape(entry_name)
            ...         moments_shape = self._compute_moments_shape(shape)
            ...         self.register_buffer("min", torch.zeros(moments_shape))
            ...         self.register_buffer("max", torch.ones(moments_shape))
        """

    @property
    def fitted(self) -> bool:
        """Whether the transform has been fitted to data.

        Returns:
            True if fit() has been called and set fitted=True, False otherwise.
        """
        return self.get_buffer("_fitted").item() == 1

    @fitted.setter
    def fitted(self, value: bool) -> None:
        """Set the fitted state.

        Args:
            value: True to mark as fitted, False otherwise.
        """
        self._fitted.fill_(1 if value else 0)
