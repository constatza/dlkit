import abc

import torch
import torch.nn as nn


class Map(nn.Module):
    """
    Base class for tensor transformations.

    Subclasses must implement a forward method.
    The inverse method is optional; if provided, it must return a Maybe[torch.Tensor]:
    Some if the inverse is successful, or Nothing otherwise.
    The fit method is also optional; if present, it will be applied before forward.
    """

    apply_inverse: bool

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_inverse = True

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    def inverse_transform(self, y: torch.Tensor) -> torch.Tensor:
        """
        Optional inverse operation.

        """
        return y  # Default: returns identity


class Scaler(Map):

    def __init__(self):
        super().__init__()
        self.fitted = False

    @abc.abstractmethod
    def fit(self, data: torch.Tensor) -> None: ...

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        self.fit(data)
        return self(data)
