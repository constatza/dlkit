import abc
from collections.abc import Sequence

import torch
import torch.nn as nn
from pydantic import validate_call, ConfigDict
from loguru import logger


class Transform(nn.Module):
    """Base class for tensor transformations.

    Subclasses must implement a forward method.
    The inverse method is optional; if provided, it must return a Maybe[torch.Tensor]:
    Some if the inverse is successful, or Nothing otherwise.
    The fit method is also optional; if present, it will be applied before forward.
    """

    input_shape: tuple[int, ...] | None
    apply_inverse: bool
    _fitted: torch.Tensor
    input_shape: torch.Tensor

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(self, input_shape: Sequence[int] | torch.Size | torch.Tensor) -> None:
        """Initialize the transform.

        Args:
            input_shape (tuple[int, ...], optional): The shape of the input data INCLUDING the first which is the batch dimension.
        """
        super().__init__()
        if not input_shape:
            logger.warning("No input shape provided. Assuming input shape (1,)")
        self.apply_inverse = True
        self.register_buffer("_fitted", torch.zeros(1, requires_grad=False))
        self.register_buffer("input_shape", torch.tensor(input_shape))

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @abc.abstractmethod
    def inverse_transform(self, y: torch.Tensor) -> torch.Tensor: ...

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the transformation to the input tensor."""
        return self.forward(x)

    def fit(self, data: torch.Tensor) -> None:
        """Fit the transformation to the data.

        Args:
            data (torch.Tensor): The data to fit the transformation to.
        """
        self.fitted = True

    @property
    def fitted(self) -> bool:
        return self.get_buffer("_fitted").item() == 1

    @fitted.setter
    def fitted(self, value: bool) -> None:
        self._fitted.fill_(1 if value else 0)
