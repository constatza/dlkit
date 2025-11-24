"""Abstract base classes for transform capabilities.

This module defines capability interfaces for transforms using the ABC pattern.
These interfaces enable explicit contracts and type-safe capability checking
with isinstance().

Design Pattern: Mixin ABCs
- Single-method abstract classes composed via multiple inheritance
- Avoids complex MI argument ordering issues
- Provides runtime isinstance() checks
- Enforces explicit capability contracts

Example:
    >>> from dlkit.core.training.transforms.base import Transform
    >>> from dlkit.core.training.transforms.interfaces import IInvertibleTransform
    >>>
    >>> class MyTransform(Transform, IInvertibleTransform):
    ...     def forward(self, x: torch.Tensor) -> torch.Tensor:
    ...         return x * 2
    ...
    ...     def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
    ...         return x / 2
    >>>
    >>> transform = MyTransform()
    >>> isinstance(transform, IInvertibleTransform)  # True
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from dlkit.core.shape_specs import IShapeSpec


class IInvertibleTransform(ABC):
    """Interface for transforms that can invert (reverse) their operation.

    Transforms implementing this interface must provide an inverse_transform() method
    that reverses the forward transformation. The inverse may require state
    from the forward pass (e.g., per-sample statistics).

    Example:
        >>> class L2Normalizer(Transform, IInvertibleTransform):
        ...     def __init__(self, input_shape):
        ...         super().__init__(input_shape)
        ...         self._last_norms = None
        ...
        ...     def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...         norms = torch.norm(x, dim=-1, keepdim=True)
        ...         self._last_norms = norms
        ...         return x / (norms + 1e-8)
        ...
        ...     def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        ...         if self._last_norms is None:
        ...             raise RuntimeError("Must call forward() first")
        ...         return x * self._last_norms

    Note:
        Implementations should raise RuntimeError if inverse_transform() requires
        forward() to be called first but it hasn't been called yet.
    """

    @abstractmethod
    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Invert the transformation applied by forward().

        Args:
            x: Transformed tensor to invert.

        Returns:
            Original tensor before transformation (approximately).

        Raises:
            RuntimeError: If inverse requires forward() state but
                forward() has not been called yet.
            RuntimeError: If batch size mismatch between forward/inverse calls.
        """
        pass


class IFittableTransform(ABC):
    """Interface for transforms that must be fitted to data before use.

    Fitted transforms learn statistics from training data (e.g., mean/std,
    min/max, PCA components) during fit() and use those statistics during
    forward() and inverse_transform().

    The fitted state is accessible via the fitted property (implemented in Transform base).
    The underlying storage is a torch.Tensor buffer to enable:
    - Checkpoint persistence via register_buffer()
    - Device movement (CPU/GPU)
    - State dict serialization

    Example:
        >>> class StandardScaler(Transform, IFittableTransform, IInvertibleTransform):
        ...     def __init__(self, dim, input_shape):
        ...         super().__init__(input_shape)
        ...         self.dim = dim
        ...         self.register_buffer('mean', torch.zeros(input_shape))
        ...         self.register_buffer('std', torch.ones(input_shape))
        ...
        ...     def fit(self, data: torch.Tensor) -> None:
        ...         self.mean = data.mean(dim=self.dim, keepdim=True)
        ...         self.std = data.std(dim=self.dim, keepdim=True)
        ...         self.fitted = True  # Property setter updates _fitted buffer
        ...
        ...     def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...         if not self.fitted:
        ...             raise RuntimeError("Transform must be fitted first")
        ...         return (x - self.mean) / (self.std + 1e-8)
        ...
        ...     def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        ...         return (x * self.std) + self.mean

    Note:
        The Transform base class provides the fitted property, which internally
        uses a _fitted torch.Tensor buffer. Use self.fitted = True to mark as fitted.
    """

    @abstractmethod
    def fit(self, data: torch.Tensor) -> None:
        """Fit transform parameters to training data.

        This method computes and stores statistics needed for forward()
        and inverse_transform() transformations. After fitting, implementations
        should set self.fitted = True.

        Args:
            data: Training data tensor to compute statistics from.
                Shape varies by transform type (typically [N, ...]).

        Side Effects:
            - Sets self.fitted to True (via property setter)
            - Stores learned statistics as tensor buffers

        Example:
            >>> scaler = StandardScaler(dim=0, input_shape=(32, 64))
            >>> scaler.fit(training_data)  # Compute mean/std
            >>> scaler.fitted  # True
        """
        pass


class ISerializableTransform(ABC):
    """Interface for transforms with custom checkpoint serialization.

    Most transforms automatically serialize via nn.Module.state_dict(), but
    some require custom serialization logic (e.g., ChainedTransform with
    nested transforms, transforms with non-tensor state).

    Example:
        >>> class ChainedTransform(BaseTransform, ISerializableTransform):
        ...     def __init__(self, transforms: list[BaseTransform]):
        ...         super().__init__()
        ...         self.transforms = transforms
        ...
        ...     def to_checkpoint_dict(self) -> dict:
        ...         return {
        ...             'transform_states': [t.state_dict() for t in self.transforms],
        ...             'transform_types': [type(t).__name__ for t in self.transforms]
        ...         }
        ...
        ...     @classmethod
        ...     def from_checkpoint_dict(cls, state: dict) -> "ChainedTransform":
        ...         # Reconstruct transforms from saved state
        ...         transforms = reconstruct_transforms(state)
        ...         return cls(transforms)
    """

    @abstractmethod
    def to_checkpoint_dict(self) -> dict:
        """Serialize transform to checkpoint-compatible dictionary.

        Returns:
            Dictionary with serialized state, JSON-compatible types preferred.
            Should include all information needed to reconstruct the transform.
        """
        pass

    @classmethod
    @abstractmethod
    def from_checkpoint_dict(cls, state: dict) -> "ISerializableTransform":
        """Reconstruct transform from checkpoint dictionary.

        Args:
            state: Dictionary returned by to_checkpoint_dict().

        Returns:
            Reconstructed transform instance with restored state.
        """
        pass


class IShapeAwareTransform(ABC):
    """Interface for transforms that require shape information.

    Shape-aware transforms need to know tensor shapes to properly allocate
    buffers, validate dimensions, or configure their behavior. This interface
    follows the same pattern as ShapeAwareModel from the neural network layer.

    Transforms implementing this interface can receive shape information via:
    1. configure_shape() method (preferred, uses shape_spec system)
    2. Lazy inference from data during fit() (fallback)

    The shape_spec system provides a single source of truth for shapes throughout
    the codebase, eliminating redundant shape tracking.

    Example:
        >>> class MinMaxScaler(Transform, IFittableTransform, IShapeAwareTransform):
        ...     def __init__(self, dim: int = 0):
        ...         super().__init__()
        ...         self.dim = dim
        ...         self._shape_configured = False
        ...
        ...     def configure_shape(
        ...         self, shape_spec: IShapeSpec, entry_name: str
        ...     ) -> None:
        ...         shape = shape_spec.get_shape(entry_name)
        ...         moments_shape = self._compute_moments_shape(shape)
        ...         self.register_buffer("min", torch.zeros(moments_shape))
        ...         self.register_buffer("max", torch.ones(moments_shape))
        ...         self._shape_configured = True
        ...
        ...     def fit(self, data: torch.Tensor) -> None:
        ...         if not self._shape_configured:
        ...             # Lazy allocation from data shape
        ...             moments_shape = self._compute_moments_shape(data.shape)
        ...             self.register_buffer("min", torch.zeros(moments_shape))
        ...             self.register_buffer("max", torch.ones(moments_shape))
        ...         # ... fit logic ...

    Note:
        Shape-aware transforms should still support lazy initialization from data
        during fit() as a fallback when shape_spec is not available.
    """

    @abstractmethod
    def configure_shape(self, shape_spec: "IShapeSpec", entry_name: str) -> None:
        """Configure transform with shape information from shape_spec.

        This method is called by the factory/pipeline system to provide shape
        information before fit() or forward() is called. Transforms should use
        this to pre-allocate buffers or validate their configuration.

        Args:
            shape_spec: Shape specification containing all entry shapes.
            entry_name: Name of the entry to get shape for (e.g., "features").

        Side Effects:
            May allocate tensor buffers, validate dimensions, or store shape info.

        Raises:
            ShapeMismatchError: If shape is incompatible with transform config.

        Example:
            >>> scaler = MinMaxScaler(dim=0)
            >>> shape_spec = create_shape_spec({"features": (32, 64)})
            >>> scaler.configure_shape(shape_spec, "features")
            >>> # scaler now has pre-allocated min/max buffers of shape (1, 64)
        """
        pass
