"""Typed, positional batch container for DLKit workflows.

This module provides the Batch dataclass that replaces string-keyed dict
batches with a typed, positional structure that enables unambiguous model
dispatch and immutable enrichment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


# TODO: Investigate whether Batch should remain a distinct type or be folded
# into TensorDict-centric flows.
@dataclass(frozen=True, slots=True, kw_only=True)
class Batch:
    """Typed, positional batch container.

    Tensors are ordered by config-entry position (index 0 = first Feature entry).
    Names are NOT stored here — they live in Feature/Target config entries and
    are accessed by the wrapper via entry configs when needed for logging
    or transforms.

    Latents carry encoder outputs produced during forward (e.g. autoencoders).

    Args:
        features: Feature tensors in config-entry order.
        targets: Target tensors in config-entry order.
        latents: Encoder latent outputs (populated during predict_step).
    """

    features: tuple[Tensor, ...]
    targets: tuple[Tensor, ...]
    latents: tuple[Tensor, ...] = field(default_factory=tuple)


def _collate_batch(batch: list[Batch], *, collate_fn_map=None) -> Batch:
    """Collate a list of Batch samples into a single batched Batch.

    Registered with PyTorch's default_collate_fn_map so DataLoader can
    handle Batch objects natively.

    Args:
        batch: List of per-sample Batch objects from Dataset.__getitem__.
        collate_fn_map: PyTorch collate function map (passed by DataLoader).

    Returns:
        Single Batch with tensors stacked along the batch dimension.
    """
    return Batch(
        features=tuple(
            torch.stack([b.features[i] for b in batch]) for i in range(len(batch[0].features))
        ),
        targets=tuple(
            torch.stack([b.targets[i] for b in batch]) for i in range(len(batch[0].targets))
        ),
        latents=tuple(
            torch.stack([b.latents[i] for b in batch]) for i in range(len(batch[0].latents))
        )
        if batch[0].latents
        else (),
    )


# Register with PyTorch's collation system so DataLoader handles Batch automatically
from torch.utils.data._utils.collate import default_collate_fn_map  # noqa: E402

default_collate_fn_map[Batch] = _collate_batch
