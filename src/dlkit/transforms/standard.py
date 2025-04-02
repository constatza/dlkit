import torch
from torch import nn

from dlkit.transforms.base import Scaler


class StdScaler(Scaler):
    def __init__(self, dim: int | list[int] = None) -> None:
        super().__init__()
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        self.dim = dim or 0

    def fit(self, data: torch.Tensor) -> None:
        self.mean = torch.mean(data, dim=self.dim, keepdim=True)
        self.std = torch.std(data, dim=self.dim, keepdim=True)
        self.fitted = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def inverse_module(self) -> nn.Module:
        """
        Return a module that lazily computes the inverse transformation
        using the current min and max values at runtime.
        """

        class LazyInverse(nn.Module):
            def __init__(self, scaler: StdScaler):
                super().__init__()
                self.scaler = scaler

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if not self.scaler.fitted:
                    raise RuntimeError(
                        "Scaler has not been fitted yet. Call `fit` before using the inverse transformation."
                    )
                device = x.device  # Get the device of the input tensor
                std_scaled = self.scaler.std.to(
                    device
                )  # Move self.scaler.max to the same device
                mean_scaled = self.scaler.mean.to(device)

                return (x * std_scaled) + mean_scaled

        return LazyInverse(self)
