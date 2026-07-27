"""Shared fixtures for engine.training tests."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pytest
import torch
from lightning.pytorch import LightningModule
from torch.utils.data import DataLoader, TensorDataset


class MeanBufferFittable(LightningModule):
    """Minimal ``Fittable`` model: fits a buffer to the mean of one dataloader pass.

    Deterministic and closed-form (no epochs/optimizer/loss), mirroring the
    real-world shape ``OneShotFitExecutor`` targets (e.g. a thin-SVD basis
    fit into a ``register_buffer``).
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(1))
        self.fit_call_count = 0
        self._fitted = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.mean

    def fit(self, dataloader: Iterable[Any]) -> None:
        self.fit_call_count += 1
        batches = [batch[0] if isinstance(batch, list | tuple) else batch for batch in dataloader]
        # No keepdim: must match the (1,)-shaped buffer declared in
        # __init__, or a freshly-constructed model's load_state_dict(strict=True)
        # rejects the checkpoint on reload (shape mismatch).
        self.mean = torch.cat(batches).mean(dim=0)
        self._fitted = True

    def is_fitted(self) -> bool:
        return self._fitted


@pytest.fixture
def fittable_model() -> MeanBufferFittable:
    """An unfitted ``MeanBufferFittable`` instance."""
    return MeanBufferFittable()


@pytest.fixture
def fit_dataloader() -> DataLoader:
    """A tiny deterministic dataloader: 4 rows, feature dim 1, mean == 2.5."""
    torch.manual_seed(0)
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    return DataLoader(TensorDataset(x), batch_size=4)


class LazyBufferFittable(LightningModule):
    """``Fittable`` model whose buffer is registered only inside ``fit()``.

    Unlike ``MeanBufferFittable``'s compile-time-constant ``(1,)`` buffer,
    this mirrors torchalg's ``PODCoarseningStrategy``: no ``register_buffer``
    call at all in ``__init__``, and the buffer's shape (truncated to
    ``rank`` columns) is only knowable once real data is seen in ``fit()``.
    """

    def __init__(self, rank: int) -> None:
        super().__init__()
        self._rank = rank

    def fit(self, dataloader: Iterable[Any]) -> None:
        batches = [batch[0] if isinstance(batch, list | tuple) else batch for batch in dataloader]
        data = torch.cat(batches)
        self.register_buffer("basis", data[:, : self._rank].clone())

    def is_fitted(self) -> bool:
        # Buffer presence is the source of truth (not a separate flag), same
        # as torchalg's PODCoarseningStrategy.is_fitted() — true after fit()
        # AND after a successful load_state_dict on a reconstructed instance.
        return "basis" in self._buffers and self._buffers["basis"] is not None


@pytest.fixture
def lazy_buffer_model() -> LazyBufferFittable:
    """An unfitted ``LazyBufferFittable`` instance."""
    return LazyBufferFittable(rank=2)


@pytest.fixture
def lazy_buffer_dataloader() -> DataLoader:
    """4 rows, feature dim 3 — enough to truncate to ``rank=2`` columns."""
    torch.manual_seed(0)
    x = torch.randn(4, 3)
    return DataLoader(TensorDataset(x), batch_size=4)
