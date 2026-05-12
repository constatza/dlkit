"""Fixtures for transformer block tests."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture
def transformer_input(batch_size: int) -> torch.Tensor:
    """8-channel, 16-timestep input for transformer blocks.

    Shape: (batch_size, 8, 16)
    """
    return torch.randn(batch_size, 8, 16)


@pytest.fixture
def attention_input() -> torch.Tensor:
    """(batch=3, channels=8, time=16) input for SelfAttentionBlock with permute=True."""
    return torch.randn(3, 8, 16)


@pytest.fixture
def attention_input_no_permute() -> torch.Tensor:
    """(batch=3, seq=16, embed=8) input for SelfAttentionBlock with permute=False."""
    return torch.randn(3, 16, 8)
