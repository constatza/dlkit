from __future__ import annotations

import pytest
import torch

from dlkit.domain.transforms.sample_norm import SampleNormL2


@pytest.fixture
def data_2d() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(32, 10)


@pytest.fixture
def data_3d() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(32, 8, 4)


def test_output_unit_l2_norm(data_2d: torch.Tensor) -> None:
    norm = SampleNormL2()
    out = norm(data_2d)
    norms = torch.norm(out, p=2, dim=1)
    assert torch.allclose(norms, torch.ones(data_2d.shape[0]), atol=1e-6)


def test_output_unit_l2_norm_3d(data_3d: torch.Tensor) -> None:
    norm = SampleNormL2(feature_dims=(1, 2))
    out = norm(data_3d)
    norms = torch.norm(out.reshape(data_3d.shape[0], -1), p=2, dim=1)
    assert torch.allclose(norms, torch.ones(data_3d.shape[0]), atol=1e-6)


def test_round_trip(data_2d: torch.Tensor) -> None:
    norm = SampleNormL2()
    out = norm(data_2d)
    recovered = norm.inverse_transform(out)
    assert torch.allclose(recovered, data_2d, atol=1e-6)


def test_inverse_fails_before_forward() -> None:
    norm = SampleNormL2()
    with pytest.raises(RuntimeError, match="forward"):
        norm.inverse_transform(torch.randn(4, 10))


def test_inverse_fails_on_batch_size_mismatch(data_2d: torch.Tensor) -> None:
    norm = SampleNormL2()
    norm(data_2d)
    with pytest.raises(RuntimeError, match="[Bb]atch"):
        norm.inverse_transform(torch.randn(data_2d.shape[0] + 1, data_2d.shape[1]))
