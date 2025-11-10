from collections.abc import Sequence

import torch
from torch import Tensor, Size
from loguru import logger
from torch.nn import ModuleList

from dlkit.tools.config.transform_settings import TransformSettings
from dlkit.tools.config import BuildContext, FactoryProvider
from .base import Transform
from .interfaces import IFittableTransform, IInvertibleTransform


# -------------------------------------------------------------------
# Core decoupled pipeline
# -------------------------------------------------------------------
class TransformChain(Transform, IFittableTransform, IInvertibleTransform):
    """Pipeline for chaining multiple transformations for one tensor stream.

    This class manages a sequence of transforms (e.g., scalers, normalizers, PCA),
    providing methods to fit them in order, apply them (``forward``), and apply their
    inverse transformations (``inverse_transform``). Use two instances to handle
    features and targets separately.

    Args:
        transform_settings (Sequence[TransformSettings] | ModuleList): Transform settings or
            an existing ModuleList of instantiated transforms to chain.
        input_shape (Sequence[int] | Size | torch.Tensor): Full input tensor shape including
            the batch dimension. If a tensor is provided, its shape is used.

    Example:
        Create and use a transform chain for features::

            chain = TransformChain(settings.feature_transforms, input_shape=(32, 64))
            chain.fit(x_train)  # type: torch.Tensor
            x_train_t = chain(x_train)
            x_orig = chain.inverse_transform(x_train_t)
    """

    transforms: ModuleList
    input_shape: tuple[int, ...] | Size
    transformed_shape: tuple[int, ...] | Size | None

    def __init__(
        self,
        transform_settings: Sequence[TransformSettings] | ModuleList,
        input_shape: Sequence[int] | Size | torch.Tensor,
    ) -> None:
        """
        Initialize the pipeline with a sequence of TransformSettings and an input shape.

        Args:
            transform_settings: Sequence of TransformSettings to instantiate.
            input_shape: Full input tensor shape including the batch dimension.

        Raises:
            ValueError: If input_shape or transform_settings fail Pydantic validation.
        """
        super().__init__(input_shape=input_shape)
        # Validate inputs

        # Convert input_shape to a tuple of ints

        # Initialize transforms (pure function call, imported from elsewhere)
        # initialize_transforms returns a torch.nn.ModuleList of instantiated transform modules
        if not isinstance(transform_settings, ModuleList):
            self.transforms = build_transforms(transform_settings, input_shape)
        else:
            self.transforms = transform_settings

        self.transformed_shape = None

    def fit(self, x: Tensor) -> None:
        """Fit the chain and propagate the intermediate output.

        Each transform is fitted in sequence (if it implements IFittableTransform)
        and then applied to produce the next input for the subsequent transform.

        Args:
            x: Input tensor to fit on (e.g., all training features).

        Raises:
            RuntimeError: If any transform fit() raises an error.
        """
        with torch.inference_mode():
            for transform in self.transforms:
                # If the transform is fittable (implements IFittableTransform), fit it
                if isinstance(transform, IFittableTransform):
                    transform.fit(x)
                # Apply the transform to produce the next input
                x = transform(x)

        # Mark as fitted and store the shape after all transforms
        self.fitted = True
        self.transformed_shape = tuple(x.shape)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the transform chain.

        Args:
            x (torch.Tensor): Input tensor to transform.

        Returns:
            torch.Tensor: Transformed tensor.

        Raises:
            RuntimeError: If ``fit(...)`` was not called before transform.
        """
        if not self.fitted:
            warn_unfit_pipeline()

        with torch.inference_mode():
            for transform in self.transforms:
                x = transform(x)
        return x

    def inverse_transform(self, x: Tensor) -> Tensor:
        """Apply the inverse transform chain in reverse order.

        Only transforms implementing IInvertibleTransform are inverted. Non-invertible
        transforms in the chain will raise a TypeError.

        Args:
            x: Tensor to invert (e.g., model output on transformed scale).

        Returns:
            Tensor mapped back to the original space.

        Raises:
            RuntimeError: If fit() was not called before inverse.
            TypeError: If any transform in chain doesn't implement IInvertibleTransform.
        """
        if not self.fitted:
            warn_unfit_pipeline()

        # Go through the transforms in reverse
        # !!IMPORTANT!!
        # inverse_transform() does not use inference mode, in case the loss function is computed
        # using the output of inverse_transform().
        for transform in reversed(self.transforms):
            if not isinstance(transform, IInvertibleTransform):
                raise TypeError(
                    f"Transform `{transform.__class__.__name__}` does not implement "
                    f"IInvertibleTransform and cannot be inverted. All transforms in a "
                    f"chain must be invertible to use inverse_transform()."
                )
            x = transform.inverse_transform(x)
        return x

    def inverse(self):
        """Return a new TransformChain with the transforms reversed.

        Returns:
            TransformChain: New chain with reversed transform order.
        """
        return TransformChain(
            transform_settings=self.transforms[::-1], input_shape=self.transformed_shape
        )


def warn_unfit_pipeline() -> None:
    """Log a warning when a chain is used before fitting."""
    logger.warning("Transform Chain called before fit.")


def build_transforms(
    transform_seq: Sequence[TransformSettings], input_shape: tuple[int, ...]
) -> ModuleList:
    """Instantiate each transform with shape propagation.

    Args:
        transform_seq (Sequence[TransformSettings]): List of transform settings.
        input_shape (tuple[int, ...]): Full input tensor shape including the batch dimension.

    Returns:
        ModuleList: List of initialized transform modules with propagated shapes.
    """
    # input_shape is expected to include the batch dimension already
    dummy_input = torch.zeros(input_shape)
    module_list = ModuleList()
    for transform in transform_seq:
        module = FactoryProvider.create_component(
            transform,
            BuildContext(mode="transform_chain", overrides={"input_shape": dummy_input.shape}),
        )
        dummy_input = module(dummy_input)
        module_list.append(module)

    return module_list
