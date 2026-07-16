from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from dlkit.domain.nn.primitives import MoESequential, RoutingStats, SparseMoE, TopKRouter

from .conftest import SkewPairRotation, assert_bounded_variance


class ScaleExpert(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale


def _zero_router(router: TopKRouter) -> None:
    with torch.no_grad():
        router.proj.weight.zero_()
        router.proj.bias.zero_()


def test_topk_router_returns_feature_prefix_shapes() -> None:
    router = TopKRouter(in_features=4, num_experts=3, top_k=2)
    x = torch.randn(2, 5, 4)

    decision = router(x)

    assert decision.weights.shape == (2, 5, 2)
    assert decision.expert_indices.shape == (2, 5, 2)
    assert decision.stats.router_probs.shape == (2, 5, 3)
    torch.testing.assert_close(decision.weights.sum(dim=-1), torch.ones(2, 5))


def test_topk_router_normalized_sigmoid_probabilities_sum_to_one() -> None:
    router = TopKRouter(
        in_features=4,
        num_experts=3,
        top_k=1,
        activation="normalized_sigmoid",
    )
    x = torch.randn(6, 4)

    stats = router(x).stats

    torch.testing.assert_close(stats.router_probs.sum(dim=-1), torch.ones(6))


def test_sparse_moe_matches_single_selected_expert_when_router_is_deterministic() -> None:
    router = TopKRouter(in_features=3, num_experts=2, top_k=1)
    _zero_router(router)
    with torch.no_grad():
        router.proj.bias[1] = 10.0
    x = torch.randn(4, 3)
    moe = SparseMoE(
        in_features=3,
        experts=[ScaleExpert(2.0), ScaleExpert(5.0)],
        router=router,
    )

    out = moe(x)

    torch.testing.assert_close(out, x * 5.0)


def test_sparse_moe_dense_topk_combines_all_experts_with_weights() -> None:
    router = TopKRouter(in_features=2, num_experts=2, top_k=2)
    _zero_router(router)
    x = torch.randn(3, 2)
    moe = SparseMoE(
        in_features=2,
        experts=[ScaleExpert(2.0), ScaleExpert(4.0)],
        router=router,
    )

    out = moe(x)

    torch.testing.assert_close(out, x * 3.0)


def test_sparse_moe_adds_shared_experts_and_returns_stats() -> None:
    router = TopKRouter(in_features=2, num_experts=1, top_k=1)
    _zero_router(router)
    x = torch.randn(3, 2)
    moe = SparseMoE(
        in_features=2,
        experts=[ScaleExpert(2.0)],
        shared_experts=[ScaleExpert(3.0)],
        shared_expert_scale=1.0,
    )
    moe.router = router

    out, stats = moe(x, return_stats=True)

    torch.testing.assert_close(out, x * 5.0)
    assert isinstance(stats, RoutingStats)
    assert stats.expert_counts.tolist() == [3]
    assert stats.aux_loss.ndim == 0


def test_sparse_moe_defaults_shared_expert_scale() -> None:
    moe = SparseMoE(
        in_features=2,
        experts=[ScaleExpert(2.0)],
        shared_experts=[ScaleExpert(3.0), ScaleExpert(4.0)],
    )

    assert moe._shared_expert_scale == pytest.approx(1 / torch.sqrt(torch.tensor(3.0)).item())


@pytest.mark.parametrize("num_layers", [8, 16, 32, 64])
def test_moe_sequential_scaled_stack_stays_bounded(num_layers: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(4096, 16)
    stack = MoESequential(
        *[
            SparseMoE(in_features=16, experts=[SkewPairRotation()], top_k=1)
            for _ in range(num_layers)
        ]
    )

    assert_bounded_variance(stack, x)


@pytest.mark.parametrize("num_layers", [8, 16, 32, 64])
def test_moe_sequential_unscaled_stack_diverges(num_layers: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(4096, 16)
    stack = MoESequential(
        *[
            SparseMoE(in_features=16, experts=[SkewPairRotation()], top_k=1)
            for _ in range(num_layers)
        ],
        branch_scale=1.0,
    )

    with torch.no_grad():
        out = stack(x)

    assert (out.std() / x.std()).item() > 2.0


def test_sparse_moe_backpropagates_to_router_and_selected_expert() -> None:
    torch.manual_seed(0)
    moe = SparseMoE(
        in_features=3,
        experts=[nn.Linear(3, 3), nn.Linear(3, 3)],
        top_k=1,
    )
    x = torch.randn(8, 3)

    loss = moe(x).sum()
    loss.backward()

    assert moe.router.proj.weight.grad is not None
    assert any(param.grad is not None for expert in moe.experts for param in expert.parameters())


def test_sparse_moe_empty_input_preserves_expert_output_width() -> None:
    moe = SparseMoE(
        in_features=3,
        experts=[nn.Linear(3, 5), nn.Linear(3, 5)],
        top_k=1,
    )
    x = torch.empty(0, 3)

    out = moe(x)

    assert out.shape == (0, 5)
