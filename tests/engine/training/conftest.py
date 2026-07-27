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
