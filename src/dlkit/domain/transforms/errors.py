"""Exception hierarchy for transform operations.

This module provides a clear exception hierarchy for transform-related errors,
enabling specific error handling and better debugging.

Design Pattern: Exception Hierarchy
- All transform errors inherit from TransformError
- Specific exception types for different failure modes
- Clear error messages with context
"""

from dlkit.common.errors import DLKitError


class TransformError(DLKitError):
    """Base exception for all transform-related errors.

    Use specific subclasses for different error types when possible.
    """


class TransformNotFittedError(TransformError):
    """Raised when a fittable transform is used before being fitted.

    Example:
        >>> scaler = MinMaxScaler(dim=0)
        >>> scaler(data)  # Raises TransformNotFittedError
    """

    def __init__(self, transform_name: str = "Transform") -> None:
        """Initialize error with transform name.

        Args:
            transform_name: Name of the transform that wasn't fitted.
        """
        super().__init__(f"{transform_name} must be fitted before use. Call fit() first.")


class ShapeMismatchError(TransformError):
    """Raised when tensor shapes are incompatible with transform requirements.

    Example:
        >>> pca = PCA(n_components=10)
        >>> pca.configure_shape(shape_spec, "features")  # features shape is (32, 5)
        >>> # Raises ShapeMismatchError: n_components > input features
    """

    def __init__(
        self, expected: tuple[int, ...], actual: tuple[int, ...], context: str = ""
    ) -> None:
        """Initialize error with shape information.

        Args:
            expected: Expected tensor shape.
            actual: Actual tensor shape received.
            context: Additional context about where mismatch occurred.
        """
        msg = f"Shape mismatch: expected {expected}, got {actual}"
        if context:
            msg += f" ({context})"
        super().__init__(msg)


class TransformChainError(TransformError):
    """Raised when a transform chain operation fails.

    Wraps errors from individual transforms with chain context.

    Example:
        >>> chain = TransformChain([scaler, pca])
        >>> chain(data)  # If scaler fails, TransformChainError wraps it
    """

    def __init__(self, transform_index: int, transform_name: str, cause: Exception) -> None:
        """Initialize error with chain context.

        Args:
            transform_index: Index of the failing transform in the chain.
            transform_name: Name/type of the failing transform.
            cause: Original exception that was raised.
        """
        super().__init__(
            f"Transform chain failed at index {transform_index} ({transform_name}): {cause}"
        )


class TransformApplicationError(TransformError):
    """Raised when applying a transform fails unexpectedly.

    Used for unexpected errors during forward() or inverse_transform().

    Example:
        >>> transform(data)  # Unexpected CUDA error
        >>> # Raises TransformApplicationError with original error wrapped
    """


class InvalidTransformConfigurationError(TransformError):
    """Raised when transform configuration is invalid.

    Example:
        >>> PCA(n_components=-5)  # Raises InvalidTransformConfigurationError
    """


class TransformAmbiguityError(TransformError):
    """Raised when multiple transforms are available but no unambiguous selection exists.

    This occurs during inference when:
    - Multiple target transforms exist (e.g., 'rhs' and 'sol')
    - Model returns a single tensor (not a dict with keys)
    - No explicit transform name is provided to disambiguate

    Example:
        >>> # Model trained with transforms for 'rhs' and 'sol'
        >>> # At inference, model returns single tensor
        >>> predictor.predict(data)  # Raises TransformAmbiguityError
        >>> # Solution: use dict output or add transform selection logic
    """

    def __init__(self, available_transforms: list[str], context: str = "") -> None:
        """Initialize error with available transform names.

        Args:
            available_transforms: List of transform names that are available.
            context: Additional context about where ambiguity occurred.
        """
        msg = (
            f"Multiple target transforms available {available_transforms}, "
            f"but cannot determine which to apply"
        )
        if context:
            msg += f" ({context})"
        msg += (
            ". Model must return dict with target names, or only one target transform should exist."
        )
        super().__init__(msg)
