from dlkit.transforms.base import Scaler, Lambda

from torch import nn
import torch


class MinMaxScaler(Scaler):

    def __init__(self, dim: int | list[int] = None) -> None:
        """Minimum-Maximum Scaler.
        Important: This scaler transforms data in the range [-1, 1] by default!!!

        Args:
            dim (int | list[int], optional): _description_. Defaults to None.
        """

        super().__init__()
        self.min: torch.Tensor | None = None
        self.max: torch.Tensor | None = None
        self.dim = dim or 0

    def fit(self, data: torch.Tensor) -> None:
        """
        Compute the minimum and maximum values along the specified dimensions of the data.

        This method adjusts the internal state of the scaler to store the minimum
        and maximum values found in the provided tensor, enabling subsequent scaling
        operations.

        Args:
            data (torch.Tensor): Input data used to determine the min and max values
                for scaling. The dimensions to be considered can be specified during
                initialization.

        """
        self.min = data.amin(dim=self.dim, keepdim=True)
        self.max = data.amax(dim=self.dim, keepdim=True)
        self.fitted = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale to interval [-1, 1]"""
        return (2 * (x - self.min) / (self.max - self.min)) - 1

    def inverse_module(self) -> nn.Module:
        """
        Return a module that lazily computes the inverse transformation
        using the current min and max values at runtime.
        """

        class LazyInverse(nn.Module):
            def __init__(self, scaler: MinMaxScaler):
                super().__init__()
                self.scaler = scaler

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if not self.scaler.fitted:
                    raise RuntimeError(
                        "Scaler has not been fitted yet. Call `fit` before using the inverse transformation."
                    )
                device = x.device  # Get the device of the input tensor
                max_scaled = self.scaler.max.to(
                    device
                )  # Move self.scaler.max to the same device
                min_scaled = self.scaler.min.to(device)

                return (x + 1) / 2 * (max_scaled - min_scaled) + min_scaled

        return LazyInverse(self)


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
