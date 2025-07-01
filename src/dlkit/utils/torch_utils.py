import torch
from collections.abc import Sequence
from torch.utils.data import DataLoader


def dataloader_to_xy(dataloader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    """Loops through a dataloader and returns the x and targets as tensors.

    Args:
        dataloader (DataLoader): The dataloader to convert.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The x and targets as tensors.
    """
    x_tensors = []
    y_tensors = []
    for batch in dataloader:
        x, y = xy_from_batch(batch)
        x_tensors.append(x)
        y_tensors.append(y)
    return torch.cat(x_tensors), torch.cat(y_tensors)


def xy_from_batch(batch):
    if isinstance(batch, dict):
        x = batch.get("features") or batch.get("x")
        y = batch.get("targets") or batch.get("y")
        return x, y
    if isinstance(batch, Sequence):
        x = batch[0]
        y = batch[1]
        return x, y
    return batch


def split_first_from_sequence(
    tensors: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Splits a sequence of predictions into the first element and the rest."""
    if isinstance(tensors, torch.Tensor):
        return tensors, ()
    return tensors[0], tuple(pred for pred in tensors[1:])
