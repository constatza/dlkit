from collections.abc import Sequence, Callable
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from loguru import logger
from torch.nn import ModuleList

from dlkit.tools.config.transform_settings import TransformSettings
from dlkit.tools.config import BuildContext, FactoryProvider
from .base import (
    Transform,
    FittableTransform,
    IncrementalFittableTransform,
    InvertibleTransform,
    ShapeAwareTransform,
)
from .errors import TransformNotFittedError, TransformChainError

if TYPE_CHECKING:
    from dlkit.core.shape_specs import IShapeSpec


# -------------------------------------------------------------------
# Shape-Aware Transform Composition (Monadic Pattern)
# -------------------------------------------------------------------


# TODO: Investigate whether ShapedTransform should remain separate or be
# expressed through TensorDict-aware transform metadata.
@dataclass(frozen=True, slots=True, kw_only=True)
class ShapedTransform:
    """A transform with its I/O shapes — the monadic unit.

    Immutable composition unit that tracks input and output shapes through
    a transform chain. Enables analytical shape inference without dummy execution.

    Attributes:
        transform: Transform instance.
        in_shape: Input tensor shape (excluding batch dimension).
        out_shape: Output tensor shape (excluding batch dimension).
    """

    transform: Any
    in_shape: tuple[int, ...]
    out_shape: tuple[int, ...]


def build_shaped_chain(
    transforms: tuple[Any, ...],
    initial_shape: tuple[int, ...],
) -> tuple[ShapedTransform, ...]:
    """Thread shapes through a transform sequence analytically.

    Each transform must implement infer_output_shape(in_shape) -> tuple[int, ...].
    Pure function — no side effects, no state mutations.

    Args:
        transforms: Ordered transform instances to compose.
        initial_shape: Shape entering the first transform (excluding batch dim).

    Returns:
        Tuple of ShapedTransform, each knowing its exact I/O shapes.

    Example:
        >>> minmax = MinMaxScaler(dim=0)
        >>> pca = PCA(n_components=10)
        >>> chain = build_shaped_chain((minmax, pca), (64,))
        >>> # chain[0] = ShapedTransform(minmax, (64,), (64,))
        >>> # chain[1] = ShapedTransform(pca, (64,), (10,))
    """

    def _bind(
        acc: tuple[tuple[ShapedTransform, ...], tuple[int, ...]],
        t: Any,
    ) -> tuple[tuple[ShapedTransform, ...], tuple[int, ...]]:
        steps, current = acc
        out = t.infer_output_shape(current)
        return steps + (
            ShapedTransform(transform=t, in_shape=current, out_shape=out),
        ), out

    steps, _ = reduce(_bind, transforms, ((), initial_shape))
    return steps


def apply_shaped_chain(x: Tensor, chain: tuple[ShapedTransform, ...]) -> Tensor:
    """Apply chain left-fold. Pure computation.

    Args:
        x: Input tensor to transform.
        chain: Sequence of ShapedTransform.

    Returns:
        Transformed tensor.
    """
    return reduce(lambda acc, step: step.transform(acc), chain, x)


def chain_output_shape(chain: tuple[ShapedTransform, ...]) -> tuple[int, ...] | None:
    """Final output shape of the chain. Pure function.

    Args:
        chain: Sequence of ShapedTransform.

    Returns:
        Final output shape or None if chain is empty.
    """
    return chain[-1].out_shape if chain else None


# -------------------------------------------------------------------
# Core decoupled pipeline
# -------------------------------------------------------------------
class TransformChain(Transform):
    """Pipeline for chaining multiple transformations for one tensor stream.

    This class manages a sequence of transforms (e.g., scalers, normalizers, PCA),
    providing methods to fit them in order, apply them (``forward``), and apply their
    inverse transformations (``inverse_transform``).

    The chain uses analytical shape inference (pure functions) for efficiency, with
    optional dummy tensor validation for compatibility checking.

    Example:
        >>> # Create chain with analytical shape inference
        >>> chain = TransformChain(transform_settings, shape_spec=shape_spec, entry_name="features")
        >>> chain.fit(x_train)
        >>> x_transformed = chain(x_train)
        >>> x_orig = chain.inverse_transform(x_transformed)
    """

    transforms: ModuleList
    transformed_shape: tuple[int, ...] | None
    _shape_spec: "IShapeSpec | None"
    _entry_name: str | None

    def __init__(
        self,
        transform_settings: Sequence[TransformSettings] | ModuleList,
        shape_spec: "IShapeSpec | None" = None,
        entry_name: str | None = None,
        validate_execution: bool = False,
    ) -> None:
        """Initialize the transform chain.

        Args:
            transform_settings: Sequence of TransformSettings to instantiate,
                or an existing ModuleList of transforms.
            shape_spec: Optional shape specification for transforms.
            entry_name: Optional entry name to look up shape in shape_spec.
            validate_execution: If True, execute dummy tensors to validate
                transform compatibility. Default False (use analytical inference).

        Example:
            >>> # Analytical inference (recommended)
            >>> chain = TransformChain(settings, shape_spec=spec, entry_name="features")
            >>>
            >>> # With validation (slower but validates compatibility)
            >>> chain = TransformChain(
            ...     settings, shape_spec=spec, entry_name="features", validate_execution=True
            ... )
        """
        super().__init__()
        self._shape_spec = shape_spec
        self._entry_name = entry_name
        self.transformed_shape = None

        # Build transforms with shape inference
        if not isinstance(transform_settings, ModuleList):
            self.transforms, self.transformed_shape = build_transforms(
                transform_settings,
                shape_spec=shape_spec,
                entry_name=entry_name,
                validate_execution=validate_execution,
            )
        else:
            self.transforms = transform_settings

    def fit(self, x: Tensor) -> None:
        """Fit the chain and propagate the intermediate output.

        Each transform is fitted in sequence (if it implements FittableTransform Protocol)
        and then applied to produce the next input for the subsequent transform.

        Uses FittableTransform Protocol for type-safe capability checking.

        Args:
            x: Input tensor to fit on (e.g., all training features).

        Raises:
            TransformChainError: If any transform fit() raises an error.
        """
        for i, transform in enumerate(self.transforms):
            try:
                # Protocol check: isinstance() with FittableTransform
                if isinstance(transform, FittableTransform):
                    transform.fit(x)
                # Apply the transform to produce the next input
                x = transform(x)
            except Exception as e:
                raise TransformChainError(
                    transform_index=i,
                    transform_name=transform.__class__.__name__,
                    cause=e,
                ) from e

        # Mark as fitted and store the shape after all transforms
        self.fitted = True
        self.transformed_shape = tuple(x.shape)

    def fit_from_dataloader(
        self,
        dataloader: Any,
        tensor_selector: Callable[[Any], Tensor],
    ) -> None:
        """Fit the chain from a re-iterable dataloader without full-data buffering.

        Each fittable transform is handled in order:
        - Incremental-capable transforms are fitted by streaming batches.
        - Non-incremental fittable transforms must already be fitted, otherwise
          fitting fails fast.

        Args:
            dataloader: Re-iterable training dataloader.
            tensor_selector: Function that extracts this chain's tensor from one batch.

        Raises:
            ValueError: If dataloader is empty.
            TypeError: If an unfitted non-incremental transform is encountered.
            TransformChainError: If any fit/apply step fails.
        """
        sample: Tensor | None = None
        for batch in dataloader:
            sample = tensor_selector(batch)
            break

        if sample is None:
            raise ValueError("Cannot fit TransformChain on empty dataloader.")

        for i, transform in enumerate(self.transforms):
            if not isinstance(transform, FittableTransform):
                continue

            if not isinstance(transform, IncrementalFittableTransform):
                if not getattr(transform, "fitted", False):
                    raise TypeError(
                        f"Incremental fitting for '{transform.__class__.__name__}' is not "
                        "implemented. Remove this transform from online fit path. "
                        "TODO: incremental PCA."
                    )
                continue

            try:
                logger.info(
                    "Incremental fit pass for transform '{}' (entry='{}', index={})",
                    transform.__class__.__name__,
                    self._entry_name or "unknown",
                    i,
                )
                transform.reset_fit_state()
                for batch in dataloader:
                    x = tensor_selector(batch)
                    for prev in self.transforms[:i]:
                        x = prev(x)
                    transform.update_fit(x)
                transform.finalize_fit()
            except Exception as e:
                raise TransformChainError(
                    transform_index=i,
                    transform_name=transform.__class__.__name__,
                    cause=e,
                ) from e

        # Infer transformed shape from one sample after fitting.
        try:
            sample_out = sample
            for i, transform in enumerate(self.transforms):
                sample_out = transform(sample_out)
            self.transformed_shape = tuple(sample_out.shape)
            self.fitted = True
        except Exception as e:
            raise TransformChainError(
                transform_index=i,
                transform_name=transform.__class__.__name__,
                cause=e,
            ) from e

    def forward(self, x: Tensor) -> Tensor:
        """Apply the transform chain.

        Args:
            x: Input tensor to transform.

        Returns:
            Transformed tensor.

        Raises:
            TransformNotFittedError: If fit() was not called before forward().
            TransformChainError: If any transform application fails.
        """
        if not self.fitted:
            raise TransformNotFittedError("TransformChain")

        for i, transform in enumerate(self.transforms):
            try:
                x = transform(x)
            except Exception as e:
                raise TransformChainError(
                    transform_index=i,
                    transform_name=transform.__class__.__name__,
                    cause=e,
                ) from e
        return x

    def inverse_transform(self, x: Tensor) -> Tensor:
        """Apply the inverse transform chain in reverse order.

        Only transforms implementing InvertibleTransform Protocol are inverted.
        Uses isinstance() check for type safety and early validation.

        Args:
            x: Tensor to invert (e.g., model output on transformed scale).

        Returns:
            Tensor mapped back to the original space.

        Raises:
            TransformNotFittedError: If fit() was not called before inverse.
            TypeError: If any transform doesn't implement InvertibleTransform.
            TransformChainError: If any inverse transform fails.
        """
        if not self.fitted:
            raise TransformNotFittedError("TransformChain")

        # Go through the transforms in reverse
        # NOTE: No inference mode, in case loss function is computed using inverse output
        for i, transform in enumerate(reversed(self.transforms)):
            if not isinstance(transform, InvertibleTransform):
                raise TypeError(
                    f"Transform `{transform.__class__.__name__}` does not implement "
                    f"InvertibleTransform Protocol and cannot be inverted. All transforms "
                    f"in a chain must be invertible to use inverse_transform()."
                )
            try:
                x = transform.inverse_transform(x)
            except Exception as e:
                raise TransformChainError(
                    transform_index=len(self.transforms) - 1 - i,
                    transform_name=transform.__class__.__name__,
                    cause=e,
                ) from e
        return x

    def inverse(self) -> "TransformChain":
        """Return a new TransformChain with the transforms reversed.

        Returns:
            New chain with reversed transform order.
        """
        return TransformChain(
            transform_settings=self.transforms[::-1],
            shape_spec=self._shape_spec,
            entry_name=self._entry_name,
        )


def build_transforms(
    transform_seq: Sequence[TransformSettings],
    shape_spec: "IShapeSpec | None" = None,
    entry_name: str | None = None,
    validate_execution: bool = False,
) -> tuple[ModuleList, tuple[int, ...] | None]:
    """Instantiate transforms with analytical shape inference or validation.

    This function uses pure analytical shape inference by default for efficiency.
    Optionally, dummy tensor execution can be used to validate transform compatibility.

    Args:
        transform_seq: List of transform settings to instantiate.
        shape_spec: Optional shape specification for initial shape.
        entry_name: Optional entry name to look up in shape_spec.
        validate_execution: If True, execute dummy tensors for validation.

    Returns:
        Tuple of (ModuleList of transforms, final output shape or None).

    Example:
        >>> # Analytical inference (fast)
        >>> transforms, output_shape = build_transforms(
        ...     settings, shape_spec=spec, entry_name="features"
        ... )
        >>>
        >>> # With validation (slower)
        >>> transforms, output_shape = build_transforms(
        ...     settings, shape_spec=spec, entry_name="features", validate_execution=True
        ... )
    """
    # Get initial shape from shape_spec
    current_shape = None
    if shape_spec and entry_name:
        current_shape = shape_spec.get_shape(entry_name)

    module_list = ModuleList()

    # Optional: Create dummy tensor for validation
    dummy_input = None
    if validate_execution and current_shape is not None:
        dummy_input = torch.zeros(current_shape)
        logger.debug(f"Transform chain validation mode enabled with shape {current_shape}")

    for transform_settings in transform_seq:
        # Create build context (no input_shape override needed)
        context = BuildContext(
            mode="transform_chain",
            shape_spec=shape_spec,
            entry_name=entry_name,
        )

        # Instantiate transform
        module = FactoryProvider.create_component(transform_settings, context)

        # Analytical shape inference using instance method (always computed for tracking)
        if current_shape is not None and hasattr(module, "infer_output_shape"):
            current_shape = module.infer_output_shape(current_shape)

        # Optional: Validate with dummy execution
        if validate_execution and dummy_input is not None:
            dummy_input = module(dummy_input)
            if current_shape is not None:
                assert tuple(dummy_input.shape) == current_shape, (
                    f"Shape mismatch: analytical={current_shape}, "
                    f"execution={tuple(dummy_input.shape)}"
                )

        module_list.append(module)

    return module_list, current_shape
