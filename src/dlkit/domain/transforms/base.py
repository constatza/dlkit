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
from collections.abc import Callable, Sequence
from functools import wraps
from typing import Protocol, final, runtime_checkable

import torch
from torch import Tensor, nn

from dlkit.common.geometry import GeometrySpec


def reshaper2d(func):
    """Decorator that handles ND inputs for transforms operating on the last dimension.

    Reshapes (..., D) → (N, D), applies func, then reshapes output back.

    Args:
        func (Callable[..., torch.Tensor]): Transform method operating on 2D tensors (N, D).

    Returns:
        Callable[..., torch.Tensor]: Wrapped function that transparently handles
            arbitrary leading dimensions.

    Example:
        >>> class MyTransform(Transform):
        ...     @reshaper2d
        ...     def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...         return x @ self.weight.T
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        data = args[-1]
        shape = data.shape
        if len(shape) < 2:
            return func(*args, **kwargs)
        data = data.reshape(-1, shape[-1])
        processed = func(*args[:-1], data, **kwargs)
        return processed.reshape((-1,) + shape[1:-1] + processed.shape[1:])

    return wrapper


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

    @property
    def fitted(self) -> bool:
        """Whether the transform has been fitted to data."""
        ...

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

    def configure_shape(self, shape_spec: GeometrySpec, entry_name: str) -> None:
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

    _fitted: bool

    def __init__(self) -> None:
        """Initialize the transform.

        Note:
            Shape information is no longer passed to __init__(). Shape-aware transforms
            should override configure_shape() and receive shapes from the shape_spec system,
            or allocate buffers lazily during fit() from data.shape.
        """
        super().__init__()
        self._fitted = False

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

    def configure_shape(self, shape_spec: GeometrySpec, entry_name: str) -> None:
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
        return self._fitted

    @fitted.setter
    def fitted(self, value: bool) -> None:
        """Set the fitted state.

        Args:
            value: True to mark as fitted, False otherwise.
        """
        self._fitted = value

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Override state_dict to include _fitted bool in the checkpoint.

        Args:
            destination: Dictionary to accumulate state dict entries.
            prefix: Prefix for parameter names.
            keep_vars: Whether to keep variables (for nn.Module compatibility).

        Returns:
            State dictionary including _fitted bool.
        """
        state = super().state_dict(
            destination=destination or {}, prefix=prefix, keep_vars=keep_vars
        )
        # Add _fitted as a plain Python bool (not a tensor)
        state[f"{prefix}_fitted"] = self._fitted
        return state

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor | bool],
        prefix: str,
        local_metadata: dict[str, int],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Override _load_from_state_dict to restore _fitted bool from checkpoint.

        Args:
            state_dict: Full state dictionary.
            prefix: Module prefix for this module's keys.
            local_metadata: Local metadata dict.
            strict: Whether to enforce strict key matching.
            missing_keys: List to accumulate missing key names.
            unexpected_keys: List to accumulate unexpected key names.
            error_msgs: List to accumulate error messages.
        """
        fitted_key = f"{prefix}_fitted"
        if fitted_key in state_dict:
            self._fitted = bool(state_dict.pop(fitted_key))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


class PartialTransform(Transform):
    """Abstract base for transforms applied to a selected subset of feature indices.

    Subclasses implement only `_compute` and `_inverse_compute` on the (already-sliced)
    tensor; scatter/gather bookkeeping is handled here. When `indices` is None the
    transform is applied to the whole tensor with no copy overhead.

    Args:
        indices: Positions along `index_dim` to transform. None applies to all.
        index_dim: Axis that holds the feature dimension. Defaults to -1 (last axis),
            which handles (N, D), (N, T, D), (N, T, Q, D) uniformly.

    Example:
        >>> class LogTransform(PartialTransform):
        ...     def _compute(self, x):
        ...         return torch.log(x + 1.0)
        ...
        ...     def _inverse_compute(self, x):
        ...         return torch.exp(x) - 1.0
        >>>
        >>> t = LogTransform(indices=[0, 3], index_dim=-1)
        >>> y = t(data)  # only features 0 and 3 are log-transformed
        >>> x = t.inverse_transform(y)  # reconstructed
    """

    def __init__(
        self,
        *,
        indices: Sequence[int] | None = None,
        index_dim: int = -1,
    ) -> None:
        """Initialize PartialTransform.

        Args:
            indices: Feature indices to transform. None means all features.
            index_dim: Axis along which to index features.
        """
        super().__init__()
        self.indices: tuple[int, ...] | None = tuple(indices) if indices is not None else None
        self.index_dim = index_dim

    @abstractmethod
    def _compute(self, x: Tensor) -> Tensor:
        """Apply the core transformation to the selected (or full) tensor.

        Args:
            x: Slice of the input tensor containing only the selected features.

        Returns:
            Transformed tensor with the same shape as x.
        """
        ...

    @abstractmethod
    def _inverse_compute(self, x: Tensor) -> Tensor:
        """Apply the inverse of the core transformation.

        Args:
            x: Slice of the transformed tensor containing only the selected features.

        Returns:
            Reconstructed tensor with the same shape as x.
        """
        ...

    def _scatter_compute(self, x: Tensor, fn: Callable[[Tensor], Tensor]) -> Tensor:
        if self.indices is None:
            return fn(x)
        dim = self.index_dim % x.ndim
        idx = torch.tensor(self.indices, device=x.device, dtype=torch.long)
        selected = torch.index_select(x, dim, idx)
        out = x.clone()
        out.index_copy_(dim, idx, fn(selected))
        return out

    @final
    def forward(self, x: Tensor) -> Tensor:
        """Apply transform to selected (or all) features.

        Args:
            x: Input tensor of any shape.

        Returns:
            Tensor with the same shape as x, with selected features transformed.
        """
        return self._scatter_compute(x, self._compute)

    @final
    def inverse_transform(self, x: Tensor) -> Tensor:
        """Invert the transform on selected (or all) features.

        Args:
            x: Transformed tensor of any shape.

        Returns:
            Tensor with the same shape as x, with selected features reconstructed.
        """
        return self._scatter_compute(x, self._inverse_compute)
