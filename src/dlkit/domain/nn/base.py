import torch
from torch import nn


class DLKitModel(nn.Module):
    """Minimal base for DLKit models — provides only the dtype property.

    Models are plain PyTorch modules with PyTorch-standard constructor args
    (in_features/out_features for linear, in_channels/in_length for conv).
    """

    @property
    def dtype(self) -> torch.dtype:
        """Infer dtype from first parameter (Lightning pattern).

        Returns:
            The dtype of the model's first parameter.

        Raises:
            RuntimeError: If the model has no parameters.
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            raise RuntimeError(
                f"{self.__class__.__name__} has no parameters, cannot determine dtype"
            )
