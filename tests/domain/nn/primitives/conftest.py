import pytest
import torch


@pytest.fixture
def basic_input():
    return torch.randn(2, 4, 16)
