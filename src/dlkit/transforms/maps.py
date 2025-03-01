import torch
from torch import nn

from dlkit.transforms.base import Invertible, Map


epsilon = 1e-8


class Log1pSigned(Invertible):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the signed logarithm with base e (natural logarithm) to the absolute value of the input.

        That is, the function maps x to sign(x) * ln(|x| + 1), where ln denotes the natural logarithm.
        """
        return torch.log1p(torch.abs(x)) * torch.sign(x)

    def inverse_module(self) -> nn.Module:
        """
        The inverse of the `direct` transformation.

        That is, the function maps y to sign(y) * (exp(|y|) - 1), where exp denotes the natural exponential function.
        """

        return Map(lambda y: torch.sign(y) * (torch.expm1(torch.abs(y))))
