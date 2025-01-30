from dlkit.io.logging import get_logger
from torch import nn
import torch

logger = get_logger(__name__)


class TransformationChain(nn.Module):
    """A pipeline that chains together a list of nn.Modules (transforms, scalers, or trainable modules).

    - Scalers have a 'fit' method that will be called once on the given data.
    - NoFitTransform does not require fit.
    - Regular nn.Modules (trainable layers) will simply pass data through.
    """

    def __init__(self, transforms: nn.ModuleList) -> None:
        """
        Args:
            transforms (List[Tuple[str, nn.Module]]):
                Ordered list of (name, module) to be executed in sequence.
        """
        super().__init__()
        # Use ModuleList so PyTorch tracks submodules
        self.direct_transforms = nn.ModuleList(step for step in transforms)
        self.inverse_transforms = nn.ModuleList(
            step.inverse() for step in transforms[::-1]
        )
        self.fitted = False

    def fit(self, data: torch.Tensor) -> None:
        """One-shot fit for all scalers in the pipeline, in order.

        Args:
            data (torch.Tensor): Data to fit scalers on.
        """
        current_data = data
        for mod in self.direct_transforms:
            # If it's a scaler, call fit
            if hasattr(mod, "fit") and callable(getattr(mod, "fit", None)):
                # Run fit
                mod.fit(current_data)
            # After optional fitting, run forward so that the data is transformed
            self.fitted = True

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

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Sequentially pass x through each step in the pipeline.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Final output after all modules.
        """
        for transform in self.inverse_transforms:
            x = transform(x)
        return x

    def to(self, device: torch.device) -> None:
        self.direct_transforms = nn.ModuleList(
            step.to(device) for step in self.direct_transforms
        )
        self.inverse_transforms = nn.ModuleList(
            step.to(device) for step in self.inverse_transforms
        )
