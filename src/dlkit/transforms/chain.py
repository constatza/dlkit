from collections.abc import Sequence

import torch
from torch import Tensor, Size
from loguru import logger
from torch.nn import ModuleList

from dlkit.settings.model_settings import TransformSettings
from dlkit.utils.loading import init_class
from .base import Transform


# -------------------------------------------------------------------
# Core decoupled pipeline
# -------------------------------------------------------------------
class TransformChain(Transform):
    """
        Pipeline for chaining multiple transformations together for ONE tensor stream.
    :w

        This class manages a sequence of transforms (e.g., scalers, normalizers, etc.),
        providing methods to fit them in order, apply them (forward), and apply their
        inverse transformations (inverse_transform). Use two instances to handle features
        and targets separately.
    """

    transforms: ModuleList
    input_shape: tuple[int, ...] | Size
    transformed_shape: tuple[int, ...] | Size | None

    def __init__(
        self,
        transform_settings: Sequence[TransformSettings] | ModuleList,
        input_shape: Sequence[int] | Size,
    ) -> None:
        """
        Initialize the pipeline with a sequence of TransformSettings and an input shape.

        Args:
            transform_settings (Sequence[Any]): A sequence of TransformSettings
                that will be turned into actual transform modules via
                `initialize_transforms(...)`.
            input_shape (Sequence[int]): The shape of the tensor that each transform
                expects as input, e.g. the feature‐dimension shape or target‐dimension shape.

        Raises:
            ValueError: If input_shape or transform_settings fail Pydantic validation.
        """
        super().__init__()
        # Validate inputs

        # Convert input_shape to a tuple of ints

        # Initialize transforms (pure function call, imported from elsewhere)
        # initialize_transforms returns a torch.nn.ModuleList of instantiated transform modules
        if not isinstance(transform_settings, ModuleList):
            self.transforms = build_transforms(transform_settings, input_shape=input_shape)
        else:
            self.transforms = transform_settings

        self.input_shape = input_shape
        self.transformed_shape = None

    def fit(self, x: Tensor) -> None:
        """
        Fit each transform in sequence (if it has a `.fit(...)` method), and then
        apply it to produce the next input for the subsequent transform.

        This is a one‐shot fit: transforms are fitted in order, and the output of
        each is fed into the next.

        Args:
            x (Tensor): The tensor to fit on (e.g., features or targets).

        Raises:
            RuntimeError: If any transform.fit raises an error.
        """
        # Record the original shape before any transforms

        with torch.no_grad():
            for transform in self.transforms:
                # If the transform supports fitting, call .fit(...)
                if hasattr(transform, "fit") and callable(transform.fit):
                    transform.fit(x)
                # Regardless, apply the transform to produce the next input
                x = transform(x)

        # Mark as fitted and store the shape after all transforms
        self.fitted = True
        self.transformed_shape = tuple(x.shape[1:])

    def forward(self, x: Tensor) -> Tensor:
        """
        Sequentially apply all transforms (in the same order used for fitting).

        Args:
            x (Tensor): Input tensor to transform.

        Returns:
            Tensor: The transformed tensor after applying all modules.

        Raises:
            RuntimeError: If `fit(...)` was not called before transform.
        """
        if not self.fitted:
            warn_unfit_pipeline()

        with torch.no_grad():
            for transform in self.transforms:
                x = transform(x)
        return x

    def inverse_transform(self, x: Tensor) -> Tensor:
        """
        Sequentially apply the inverse transformations in reverse order.

        Args:
            x (Tensor): Tensor to invert (e.g., model output on transformed scale).

        Returns:
            Tensor: The tensor mapped back to the original space.

        Raises:
            RuntimeError: If `fit(...)` was not called before inverse.
            AttributeError: If any transform lacks `inverse_transform(...)`.
        """
        if not self.fitted:
            warn_unfit_pipeline()

        with torch.no_grad():
            # Go through the transforms in reverse
            for transform in reversed(self.transforms):
                if not hasattr(transform, "inverse_transform") or not callable(
                    transform.inverse_transform
                ):
                    raise AttributeError(
                        f"Transform `{transform.__class__.__name__}` does not support inverse_transform."
                    )
                x = transform.inverse_transform(x)
        return x

    def inverse(self):
        """Returns a new TransformChain with the transforms reversed."""
        return TransformChain(
            transform_settings=self.transforms[::-1], input_shape=self.transformed_shape
        )


def warn_unfit_pipeline() -> None:
    error = "Pipeline must be fitted before calling inverse_transform."
    logger.warning(error)


def build_transforms(
    transform_seq: Sequence[TransformSettings], input_shape: tuple[int, ...] | None = None
) -> ModuleList:
    """Initialize the transforms in the pipeline and provide shape information to each transform.

    Args:
        transform_seq (Sequence[TransformSettings]): List of transform settings.
        input_shape (tuple[int, ...], optional): Shape of the input data. Defaults to None.

    Returns:
                nn.ModuleList: List of initialized transform modules.
    """
    dummy_input = torch.zeros((1, *input_shape))
    module_list = ModuleList()
    for transform in transform_seq:
        module = init_class(transform, input_shape=input_shape)
        dummy_input = module(dummy_input)
        module_list.append(module)
        input_shape = dummy_input.shape[1:]

    return module_list
