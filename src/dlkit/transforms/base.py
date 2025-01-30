import abc
from typing import Callable

import torch
import torch.nn as nn


class Lambda(nn.Module):
    def __init__(self, func: Callable):
        super().__init__()
        self.func = func

    def forward(self, x):
        with torch.no_grad():
            return self.func(x)


class Invertible(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def inverse(self) -> nn.Module:
        pass

    def direct(self, x: torch.Tensor) -> nn.Module:
        return self


class Scaler(Invertible):
    """A module (scaler) that REQUIRES a one-time fit, e.g. for computing statistics."""

    def __init__(self) -> None:
        """Initialize BaseScaler."""
        super().__init__()
        self.fitted = False

    @abc.abstractmethod
    def fit(self, data: torch.Tensor) -> None:
        """Fit the scaler on data.

        Args:
            data (torch.Tensor): Data to fit on.
        """
        pass

    def flag_fit(self) -> None:
        self.fitted = True
