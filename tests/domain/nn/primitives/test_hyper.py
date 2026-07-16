from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor, nn

from dlkit.domain.nn.primitives import (
    GraphHyperConnection,
    GraphHyperSequential,
    HyperConnection,
    HyperSequential,
    LaneExpand,
    LaneMixingStats,
    LaneReduce,
    residual_branch_scale,
)

from .conftest import GraphSkewPairRotation, SkewPairRotation, assert_bounded_variance


class ZeroModule(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.zeros_like(x)


class GraphScaleModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.edge_index_seen: Tensor | None = None
        self.edge_attr_seen: Tensor | None = None

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        self.edge_index_seen = edge_index
        self.edge_attr_seen = edge_attr
        return x * 2.0


def test_lane_expand_adds_lane_axis_before_features() -> None:
    x = torch.randn(4, 3)
    expanded = LaneExpand(3)(x)

    assert expanded.shape == (4, 3, 3)
    for lane in range(3):
        torch.testing.assert_close(expanded[:, lane, :], x)


def test_lane_reduce_is_identity_biased_to_first_lane() -> None:
    lanes = torch.randn(5, 3, 4)
    reducer = LaneReduce(3)

    out = reducer(lanes)

    torch.testing.assert_close(out, lanes[:, 0, :], rtol=1e-5, atol=1e-5)


def test_hyper_connection_with_zero_branch_starts_as_identity() -> None:
    x = torch.randn(6, 4)
    block = HyperConnection(ZeroModule(), num_lanes=3)

    out = block(x)

    torch.testing.assert_close(out, x, rtol=1e-5, atol=1e-5)


def test_hyper_connection_can_return_lanes_and_preserve_lane_shape() -> None:
    x = torch.randn(2, 4)
    block = HyperConnection(nn.Identity(), num_lanes=2, branch_scale=0.5)

    lanes = block(x, return_lanes=True)

    assert lanes.shape == (2, 2, 4)
    torch.testing.assert_close(lanes[:, 0, :], x * 1.5)
    torch.testing.assert_close(lanes[:, 1, :], x * 1.5)


def test_hyper_connection_accepts_existing_lane_tensor() -> None:
    lanes = torch.randn(2, 3, 4)
    block = HyperConnection(ZeroModule(), num_lanes=3)

    out = block(lanes)

    assert out.shape == lanes.shape
    torch.testing.assert_close(out, lanes)


def test_hyper_connection_mixing_stats_are_scalar_tensors() -> None:
    block = HyperConnection(nn.Identity(), num_lanes=3)

    stats = block.mixing_stats()

    assert isinstance(stats, LaneMixingStats)
    assert stats.lane_entropy.ndim == 0
    assert stats.dominant_lane_fraction.ndim == 0
    assert stats.off_diagonal_norm.ndim == 0


def test_graph_hyper_connection_forwards_graph_arguments_per_lane() -> None:
    module = GraphScaleModule()
    block = GraphHyperConnection(module, num_lanes=2, branch_scale=1.0)
    x = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    edge_attr = torch.randn(3, 1)

    out = block(x, edge_index, edge_attr=edge_attr)

    torch.testing.assert_close(out, x * 3.0, rtol=1e-5, atol=1e-5)
    assert module.edge_index_seen is edge_index
    assert module.edge_attr_seen is edge_attr


def test_residual_branch_scale_matches_stack_variance_normalization() -> None:
    assert residual_branch_scale(0) == 1.0
    assert residual_branch_scale(8) == pytest.approx(1 / math.sqrt(8))
    assert residual_branch_scale(8, residual_branches_per_layer=2) == pytest.approx(
        1 / math.sqrt(16)
    )


@pytest.mark.parametrize("num_layers", [8, 16, 32, 64])
def test_deep_hyper_sequential_output_std_stays_bounded(num_layers: int) -> None:
    """Output std stays bounded across depth for HyperSequential's default scale.

    The skew branch preserves per-feature variance while being non-identity, so
    an unscaled residual stack compounds geometrically. HyperSequential's default
    branch_scale=1/sqrt(num_layers) keeps that growth bounded across depths.
    """
    torch.manual_seed(0)
    stack = HyperSequential(
        *(SkewPairRotation() for _ in range(num_layers)),
        num_lanes=1,
        lane_init="identity",
    )
    x = torch.randn(64, 16)

    assert stack.branch_scale == pytest.approx(1 / math.sqrt(num_layers))
    assert_bounded_variance(stack, x)


def test_hyper_sequential_unscaled_skew_branch_variance_diverges() -> None:
    torch.manual_seed(0)
    num_layers = 16
    x = torch.randn(4096, 16)
    stack = HyperSequential(
        *(SkewPairRotation() for _ in range(num_layers)),
        num_lanes=1,
        branch_scale=1.0,
        lane_init="identity",
    )

    out = stack(x)
    std_ratio = out.std() / x.std()

    assert std_ratio.item() > 100.0


@pytest.mark.parametrize("num_layers", [8, 16, 32, 64])
def test_deep_graph_hyper_sequential_output_std_stays_bounded(num_layers: int) -> None:
    """Output std stays bounded across depth for GraphHyperSequential's default scale.

    Mirrors test_deep_hyper_sequential_output_std_stays_bounded but through the
    graph-shaped forward contract (edge_index/edge_attr are unused by the branch).
    """
    torch.manual_seed(0)
    stack = GraphHyperSequential(
        *(GraphSkewPairRotation() for _ in range(num_layers)),
        num_lanes=1,
        lane_init="identity",
    )
    x = torch.randn(64, 16)
    edge_index = torch.zeros(2, 1, dtype=torch.long)

    assert stack.branch_scale == pytest.approx(1 / math.sqrt(num_layers))
    stack.eval()
    with torch.no_grad():
        out = stack(x, edge_index)
    ratio = (out.std() / x.std()).item()
    assert 0.5 < ratio < 2.0, f"Output std ratio {ratio:.2f} outside [0.5, 2.0]"


def test_graph_hyper_sequential_unscaled_skew_branch_variance_diverges() -> None:
    torch.manual_seed(0)
    num_layers = 16
    x = torch.randn(4096, 16)
    edge_index = torch.zeros(2, 1, dtype=torch.long)
    stack = GraphHyperSequential(
        *(GraphSkewPairRotation() for _ in range(num_layers)),
        num_lanes=1,
        branch_scale=1.0,
        lane_init="identity",
    )

    out = stack(x, edge_index)
    std_ratio = out.std() / x.std()

    assert std_ratio.item() > 100.0
