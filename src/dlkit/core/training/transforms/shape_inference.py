"""Pure functions for inferring transform output shapes.

This module provides analytical shape inference without executing transforms.
Each transform type has a registered shape inference function that computes
output shape from input shape and transform parameters.

Design Pattern: Strategy Registry
- Pure functions (no side effects)
- Registry maps transform types to inference functions
- Enables fast shape propagation through transform chains
- No dummy tensor execution required

Example:
    >>> from dlkit.core.training.transforms import MinMaxScaler
    >>> infer_func = SHAPE_INFERENCE_REGISTRY[MinMaxScaler]
    >>> output_shape = infer_func(input_shape=(32, 64), dim=0)
    >>> # output_shape = (32, 64) - MinMaxScaler preserves shape
"""

from collections.abc import Callable
from typing import Any

# Type alias for shape inference functions
ShapeInferenceFunc = Callable[..., tuple[int, ...]]

# Registry mapping transform types to their shape inference functions
SHAPE_INFERENCE_REGISTRY: dict[type, ShapeInferenceFunc] = {}


def register_shape_inference(transform_cls: type) -> Callable:
    """Decorator to register a shape inference function for a transform type.

    Args:
        transform_cls: Transform class to register inference function for.

    Returns:
        Decorator function that registers the inference function.

    Example:
        >>> @register_shape_inference(MinMaxScaler)
        ... def infer_minmax_output_shape(
        ...     input_shape: tuple[int, ...], dim: int = 0, **kwargs
        ... ) -> tuple[int, ...]:
        ...     return input_shape  # Shape-preserving
    """

    def decorator(func: ShapeInferenceFunc) -> ShapeInferenceFunc:
        SHAPE_INFERENCE_REGISTRY[transform_cls] = func
        return func

    return decorator


# Shape-preserving transforms (most common case)


def infer_shape_preserving(input_shape: tuple[int, ...], **kwargs: Any) -> tuple[int, ...]:
    """Generic shape inference for shape-preserving transforms.

    Args:
        input_shape: Input tensor shape.
        **kwargs: Ignored transform parameters.

    Returns:
        Same as input_shape.
    """
    return input_shape


def infer_pca_output_shape(
    input_shape: tuple[int, ...], n_components: int, **kwargs: Any
) -> tuple[int, ...]:
    """Infer PCA output shape.

    PCA reduces the last dimension to n_components.

    Args:
        input_shape: Input tensor shape (N, D) or (N, H, W, D).
        n_components: Number of principal components to keep.
        **kwargs: Ignored transform parameters.

    Returns:
        Output shape with last dimension = n_components.

    Example:
        >>> infer_pca_output_shape((100, 64), n_components=10)
        (100, 10)
    """
    return input_shape[:-1] + (n_components,)


def infer_tensor_subset_output_shape(
    input_shape: tuple[int, ...],
    keep: list[int] | tuple[int, ...] | slice,
    dim: int = 1,
    **kwargs: Any,
) -> tuple[int, ...]:
    """Infer TensorSubset output shape.

    TensorSubset reduces the specified dimension to len(keep).

    Args:
        input_shape: Input tensor shape.
        keep: Indices to keep (list, tuple, or slice).
        dim: Dimension to subset.
        **kwargs: Ignored transform parameters.

    Returns:
        Output shape with dimension dim = len(keep).

    Example:
        >>> infer_tensor_subset_output_shape((32, 100), keep=[0, 5, 10], dim=1)
        (32, 3)
    """
    output_shape = list(input_shape)

    # Compute length from keep parameter
    if isinstance(keep, slice):
        # Handle slice objects
        start = keep.start or 0
        stop = keep.stop or input_shape[dim]
        step = keep.step or 1
        length = len(range(start, stop, step))
    else:
        # Handle lists/tuples
        length = len(keep)

    output_shape[dim] = length
    return tuple(output_shape)


def infer_permutation_output_shape(
    input_shape: tuple[int, ...], dims: tuple[int, ...], **kwargs: Any
) -> tuple[int, ...]:
    """Infer Permutation output shape.

    Permutation reorders dimensions according to dims parameter.

    Args:
        input_shape: Input tensor shape.
        dims: New dimension order (e.g., (0, 2, 1) swaps last two).
        **kwargs: Ignored transform parameters.

    Returns:
        Output shape with reordered dimensions.

    Example:
        >>> infer_permutation_output_shape((32, 64, 128), dims=(0, 2, 1))
        (32, 128, 64)
    """
    return tuple(input_shape[d] for d in dims)


def infer_output_shape(
    transform_type: type, input_shape: tuple[int, ...], **transform_kwargs: Any
) -> tuple[int, ...]:
    """Infer output shape for any registered transform.

    Args:
        transform_type: Type of the transform.
        input_shape: Input tensor shape.
        **transform_kwargs: Transform constructor/configuration parameters.

    Returns:
        Inferred output shape.

    Raises:
        KeyError: If transform_type not registered.

    Example:
        >>> from dlkit.core.training.transforms import PCA
        >>> infer_output_shape(PCA, (100, 64), n_components=10)
        (100, 10)
    """
    inference_func = SHAPE_INFERENCE_REGISTRY[transform_type]
    return inference_func(input_shape, **transform_kwargs)


# Register will be called by individual transform modules when imported
# This avoids circular imports and keeps registration decentralized
