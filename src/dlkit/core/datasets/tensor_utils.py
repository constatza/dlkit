"""Tensor utility functions for dataset operations."""

import torch


def ensure2d(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure that node- or graph-level features are at least 2D.

    Args:
        tensor (torch.Tensor): The node- or graph-level features.

    Returns:
        torch.Tensor: The tensor with at least 2 dimensions.
    """
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(1)
    return tensor
