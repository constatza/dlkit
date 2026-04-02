from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from loguru import logger
from torch import Tensor
from torch.nn import ModuleList

from .base import (
    FittableTransform,
    IncrementalFittableTransform,
    InvertibleTransform,
    Transform,
)
from .errors import TransformChainError, TransformNotFittedError

if TYPE_CHECKING:
    from dlkit.shared.shapes import ShapeSpecProtocol


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
    _shape_spec: ShapeSpecProtocol | None
    _entry_name: str | None

    def __init__(
        self,
        transforms: ModuleList,
        shape_spec: ShapeSpecProtocol | None = None,
        entry_name: str | None = None,
    ) -> None:
        """Initialize the transform chain from a pre-built ModuleList.

        Args:
            transforms: ModuleList of instantiated transform modules.
            shape_spec: Optional shape specification for transforms.
            entry_name: Optional entry name to look up shape in shape_spec.

        Note:
            TransformChain no longer builds transforms from settings. Use
            ``component_builders.build_transform_list()`` to instantiate transforms,
            then pass the ModuleList here.

        Example:
            >>> from dlkit.runtime.workflows.factories.component_builders import (
            ...     build_transform_list,
            ... )
            >>> module_list, shape = build_transform_list(settings, shape_spec=spec)
            >>> chain = TransformChain(module_list, shape_spec=spec, entry_name="features")
        """
        super().__init__()
        self._shape_spec = shape_spec
        self._entry_name = entry_name
        self.transforms = transforms
        self.transformed_shape = None

    def fit(self, data: Tensor) -> None:
        """Fit the chain and propagate the intermediate output.

        Each transform is fitted in sequence (if it implements FittableTransform Protocol)
        and then applied to produce the next input for the subsequent transform.

        Uses FittableTransform Protocol for type-safe capability checking.

        Args:
            data: Input tensor to fit on (e.g., all training features).

        Raises:
            TransformChainError: If any transform fit() raises an error.
        """
        for i, transform in enumerate(self.transforms):
            try:
                # Protocol check: use separate var so transform is not narrowed for __call__
                if isinstance(transform, FittableTransform):
                    transform.fit(data)
                data = transform(data)
            except Exception as e:
                raise TransformChainError(
                    transform_index=i,
                    transform_name=transform.__class__.__name__,
                    cause=e,
                ) from e

        # Mark as fitted and store the shape after all transforms
        self.fitted = True
        self.transformed_shape = tuple(data.shape)

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
                    for prev in list(self.transforms)[:i]:
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
        transform_index = 0
        transform = self.transforms[0] if self.transforms else None
        try:
            sample_out = sample
            for transform in self.transforms:
                sample_out = transform(sample_out)
                transform_index += 1
            self.transformed_shape = tuple(sample_out.shape)
            self.fitted = True
        except Exception as e:
            raise TransformChainError(
                transform_index=transform_index,
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

    def inverse(self) -> TransformChain:
        """Return a new TransformChain with the transforms reversed.

        Returns:
            New chain with reversed transform order.
        """
        return TransformChain(
            transforms=ModuleList(list(reversed(self.transforms))),
            shape_spec=self._shape_spec,
            entry_name=self._entry_name,
        )
