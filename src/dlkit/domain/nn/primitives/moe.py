from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

RouterActivation = Literal["softmax", "normalized_sigmoid"]
DropPolicy = Literal["none", "drop"]


@dataclass(frozen=True)
class RoutingStats:
    """Diagnostics emitted by a sparse MoE router."""

    aux_loss: Tensor
    expert_counts: Tensor
    router_probs: Tensor
    selected_experts: Tensor


@dataclass(frozen=True)
class RoutingDecision:
    """Concrete token-to-expert assignment used by ``SparseMoE``."""

    weights: Tensor
    expert_indices: Tensor
    stats: RoutingStats


def _normalize_sigmoid(logits: Tensor) -> Tensor:
    probs = torch.sigmoid(logits)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(probs.dtype).eps)


def _switch_aux_loss(probs: Tensor, expert_indices: Tensor, num_experts: int) -> Tensor:
    token_count = probs.shape[0]
    if token_count == 0:
        return probs.new_zeros(())

    selected = torch.zeros(token_count, num_experts, device=probs.device, dtype=probs.dtype)
    selected.scatter_add_(
        1,
        expert_indices,
        torch.ones_like(expert_indices, dtype=probs.dtype),
    )
    selected = selected.clamp_max(1.0)
    expert_fraction = selected.mean(dim=0)
    probability_fraction = probs.mean(dim=0)
    return num_experts * torch.sum(expert_fraction * probability_fraction)


class TopKRouter(nn.Module):
    """Feature-last top-k token router for sparse mixture-of-experts layers.

    The router is intentionally independent from the expert implementation. It
    maps each flattened token/sample to ``top_k`` expert ids plus normalized
    dispatch weights, and emits load-balancing diagnostics for training code to
    consume if desired.
    """

    def __init__(
        self,
        *,
        in_features: int,
        num_experts: int,
        top_k: int = 2,
        activation: RouterActivation = "softmax",
        bias: bool = True,
        jitter_noise: float = 0.0,
        capacity_factor: float | None = None,
        drop_policy: DropPolicy = "none",
    ) -> None:
        super().__init__()
        if in_features <= 0:
            raise ValueError("in_features must be positive")
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if activation not in ("softmax", "normalized_sigmoid"):
            raise ValueError("activation must be 'softmax' or 'normalized_sigmoid'")
        if jitter_noise < 0.0:
            raise ValueError("jitter_noise must be non-negative")
        if capacity_factor is not None and capacity_factor <= 0.0:
            raise ValueError("capacity_factor must be positive when provided")
        if drop_policy not in ("none", "drop"):
            raise ValueError("drop_policy must be 'none' or 'drop'")
        if capacity_factor is not None and drop_policy == "none":
            raise ValueError("capacity_factor requires drop_policy='drop'")

        self.in_features = in_features
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.activation = activation
        self.jitter_noise = jitter_noise
        self.capacity_factor = capacity_factor
        self.drop_policy = drop_policy
        self.proj = nn.Linear(in_features, num_experts, bias=bias)

    def forward(self, x: Tensor) -> RoutingDecision:
        """Route a feature-last tensor after flattening all non-feature axes."""
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"TopKRouter expected last dimension {self.in_features}, got {x.shape[-1]}"
            )

        flat = x.reshape(-1, self.in_features)
        logits = self.proj(flat)
        if self.training and self.jitter_noise > 0.0:
            logits = logits + torch.empty_like(logits).uniform_(
                -self.jitter_noise, self.jitter_noise
            )

        probs = torch.softmax(logits, dim=-1)
        if self.activation == "normalized_sigmoid":
            probs = _normalize_sigmoid(logits)

        top_weights, top_indices = torch.topk(probs, k=self.top_k, dim=-1)
        top_weights = self._apply_capacity(top_weights, top_indices)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(top_weights.dtype).eps
        )

        active_indices = top_indices[top_weights > 0]
        counts = torch.bincount(active_indices, minlength=self.num_experts)
        stats = RoutingStats(
            aux_loss=_switch_aux_loss(probs, top_indices, self.num_experts),
            expert_counts=counts,
            router_probs=probs.reshape(*x.shape[:-1], self.num_experts),
            selected_experts=top_indices.reshape(*x.shape[:-1], self.top_k),
        )
        return RoutingDecision(
            weights=top_weights.reshape(*x.shape[:-1], self.top_k),
            expert_indices=top_indices.reshape(*x.shape[:-1], self.top_k),
            stats=stats,
        )

    def _apply_capacity(self, weights: Tensor, expert_indices: Tensor) -> Tensor:
        if self.capacity_factor is None:
            return weights

        token_count = expert_indices.shape[0]
        capacity = max(
            1, math.ceil(self.capacity_factor * token_count * self.top_k / self.num_experts)
        )
        keep = torch.ones_like(weights, dtype=torch.bool)
        for expert_id in range(self.num_experts):
            assigned = (expert_indices == expert_id).nonzero(as_tuple=False)
            if assigned.shape[0] > capacity:
                overflow = assigned[capacity:]
                keep[overflow[:, 0], overflow[:, 1]] = False
        return weights * keep.to(dtype=weights.dtype)


class SparseMoE(nn.Module):
    """Sparse mixture-of-experts wrapper for feature-last tensors.

    Experts are ordinary modules. Each selected expert receives only its routed
    tokens, and shared experts, when provided, run for every token.
    """

    def __init__(
        self,
        *,
        in_features: int,
        experts: Sequence[nn.Module],
        top_k: int = 2,
        router: TopKRouter | None = None,
        shared_experts: Sequence[nn.Module] = (),
        router_activation: RouterActivation = "softmax",
        capacity_factor: float | None = None,
        drop_policy: DropPolicy = "none",
        jitter_noise: float = 0.0,
        return_stats: bool = False,
        shared_expert_scale: float | None = None,
    ) -> None:
        super().__init__()
        if not experts:
            raise ValueError("experts must contain at least one module")

        self.in_features = in_features
        self.experts = nn.ModuleList(experts)
        self.shared_experts = nn.ModuleList(shared_experts)
        self._shared_expert_scale = (
            1.0 / math.sqrt(1 + len(self.shared_experts))
            if shared_expert_scale is None
            else shared_expert_scale
        )
        self.return_stats = return_stats
        self.router = router or TopKRouter(
            in_features=in_features,
            num_experts=len(experts),
            top_k=top_k,
            activation=router_activation,
            capacity_factor=capacity_factor,
            drop_policy=drop_policy,
            jitter_noise=jitter_noise,
        )
        if self.router.num_experts != len(self.experts):
            raise ValueError("router.num_experts must match len(experts)")

    def forward(
        self, x: Tensor, *, return_stats: bool | None = None
    ) -> Tensor | tuple[Tensor, RoutingStats]:
        """Apply routed and shared experts to a feature-last input tensor."""
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"SparseMoE expected last dimension {self.in_features}, got {x.shape[-1]}"
            )

        original_shape = x.shape[:-1]
        flat = x.reshape(-1, self.in_features)
        decision = self.router(x)
        weights = decision.weights.reshape(-1, self.router.top_k)
        indices = decision.expert_indices.reshape(-1, self.router.top_k)

        output: Tensor | None = None
        for route_rank in range(self.router.top_k):
            route_weights = weights[:, route_rank]
            route_indices = indices[:, route_rank]
            for expert_id, expert in enumerate(self.experts):
                mask = (route_indices == expert_id) & (route_weights > 0)
                if not torch.any(mask):
                    continue
                expert_out = expert(flat[mask])
                if output is None:
                    output = flat.new_zeros(flat.shape[0], expert_out.shape[-1])
                output[mask] = output[mask] + expert_out * route_weights[mask].unsqueeze(-1)

        shared: Tensor | None = None
        if self.shared_experts:
            shared = torch.stack([expert(flat) for expert in self.shared_experts], dim=0).mean(
                dim=0
            )

        if output is None:
            if shared is not None:
                output = torch.zeros_like(shared)
            else:
                output = torch.zeros_like(self.experts[0](flat))

        if shared is not None:
            output = output + self._shared_expert_scale * shared

        out = output.reshape(*original_shape, output.shape[-1])
        should_return_stats = self.return_stats if return_stats is None else return_stats
        if should_return_stats:
            return out, decision.stats
        return out


__all__ = [
    "DropPolicy",
    "RouterActivation",
    "RoutingDecision",
    "RoutingStats",
    "SparseMoE",
    "TopKRouter",
]
