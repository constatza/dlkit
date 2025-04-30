from collections.abc import Sequence

import torch
from torch.nn import ModuleList

from .base import Scaler


class TransformationChain(Scaler):
    direct_transforms: ModuleList
    inverse_transforms: ModuleList
    fitted: bool

    def __init__(self, transforms: Sequence[torch.nn.Module]):
        super().__init__()
        # Use ModuleList so PyTorch tracks submodules
        self.direct_transforms = ModuleList(transforms)
        self.fitted = False
        self.inverse_transforms: ModuleList = ModuleList(reversed(transforms))

    def fit(self, data: torch.Tensor) -> None:
        """One-shot fit for all scalers in the pipeline, in order.

        Args:
            data (torch.Tensor): Data to fit scalers on.
        """
        current_data = data
        for mod in self.direct_transforms:
            # If it's a scaler, call fit
            if hasattr(mod, "fit") and callable(mod.fit):
                mod.fit(current_data)
            current_data = mod(current_data)
        self.fitted = True

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        """One-shot fit for all scalers in the pipeline, in order.

        Args:
            data (torch.Tensor): Data to fit scalers on.
        """
        self.fit(data)
        return self(data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sequentially pass x through each step in the pipeline.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Final output after all modules.
        """
        for transform in self.direct_transforms:
            x = transform(x)
        return x

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Sequentially pass x through each step in the pipeline.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Final output after all modules.
        """
        for transform in self.inverse_transforms:
            if not hasattr(transform, "inverse_transform"):
                raise ValueError(
                    f"Transform {transform} does not have an inverse_transform method."
                )
            if transform.apply_inverse:
                x = transform.inverse_transform(x)
        return x
