from __future__ import annotations

import math
from typing import Literal, cast

import torch
from torch import Tensor, nn

from .hyper import GraphHyperConnection, HyperConnection, LaneExpand, LaneMixingStats, LaneReduce
from .moe import RoutingStats, SparseMoE

LaneInit = Literal["identity", "asymmetric"]


def residual_branch_scale(
    num_layers: int,
    *,
    residual_branches_per_layer: int = 1,
) -> float:
    """Return branch scale for ordinary residual stacks."""

    if num_layers < 0:
        raise ValueError("num_layers must be non-negative")
    if residual_branches_per_layer <= 0:
        raise ValueError("residual_branches_per_layer must be positive")
    if num_layers == 0:
        return 1.0

    return 1.0 / math.sqrt(num_layers * residual_branches_per_layer)


def _apply_asymmetric_init(layer: HyperConnection | GraphHyperConnection) -> None:
    if layer.num_lanes <= 1:
        return
    offsets = layer.pre_delta.detach().new_tensor(
        [((i / (layer.num_lanes - 1)) - 0.5) * 1e-3 for i in range(layer.num_lanes)]
    )
    with torch.no_grad():
        layer.pre_delta.diagonal().copy_(offsets)
        layer.post_delta.diagonal().copy_(-offsets)


class HyperSequential(nn.Module):
    """Sequential stack that keeps Hyper-Connection lanes internal."""

    def __init__(
        self,
        *modules: nn.Module,
        num_lanes: int = 2,
        branch_scale: float | None = None,
        residual_branches_per_layer: int = 1,
        lane_init: LaneInit = "asymmetric",
        reduce: bool = True,
    ) -> None:
        super().__init__()
        if num_lanes <= 0:
            raise ValueError("num_lanes must be positive")
        scale = (
            residual_branch_scale(
                len(modules),
                residual_branches_per_layer=residual_branches_per_layer,
            )
            if branch_scale is None
            else branch_scale
        )
        self.num_lanes = num_lanes
        self.reduce = reduce
        self.expand = LaneExpand(num_lanes)
        self.reducer = LaneReduce(num_lanes)
        layers = [
            HyperConnection(module, num_lanes=num_lanes, branch_scale=scale) for module in modules
        ]
        if lane_init == "asymmetric":
            for layer in layers:
                _apply_asymmetric_init(layer)
        elif lane_init != "identity":
            raise ValueError("lane_init must be 'identity' or 'asymmetric'")
        self.layers = nn.ModuleList(layers)
        self.branch_scale = scale

    def forward(self, x: Tensor, *, return_lanes: bool = False) -> Tensor:
        has_lanes = x.ndim >= 3 and x.shape[-2] == self.num_lanes
        lanes = x if has_lanes else self.expand(x)
        for layer in self.layers:
            lanes = layer(lanes, return_lanes=True)
        if return_lanes or not self.reduce:
            return lanes
        return self.reducer(lanes)

    def mixing_stats(self) -> list[LaneMixingStats]:
        return [cast(HyperConnection, layer).mixing_stats() for layer in self.layers]


class GraphHyperSequential(nn.Module):
    """Graph-aware Hyper-Connection stack with a graph ``forward`` contract."""

    def __init__(
        self,
        *modules: nn.Module,
        num_lanes: int = 2,
        branch_scale: float | None = None,
        residual_branches_per_layer: int = 1,
        lane_init: LaneInit = "asymmetric",
        reduce: bool = True,
    ) -> None:
        super().__init__()
        if num_lanes <= 0:
            raise ValueError("num_lanes must be positive")
        scale = (
            residual_branch_scale(
                len(modules),
                residual_branches_per_layer=residual_branches_per_layer,
            )
            if branch_scale is None
            else branch_scale
        )
        self.num_lanes = num_lanes
        self.reduce = reduce
        self.expand = LaneExpand(num_lanes)
        self.reducer = LaneReduce(num_lanes)
        layers = [
            GraphHyperConnection(module, num_lanes=num_lanes, branch_scale=scale)
            for module in modules
        ]
        if lane_init == "asymmetric":
            for layer in layers:
                _apply_asymmetric_init(layer)
        elif lane_init != "identity":
            raise ValueError("lane_init must be 'identity' or 'asymmetric'")
        self.layers = nn.ModuleList(layers)
        self.branch_scale = scale

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        *,
        return_lanes: bool = False,
    ) -> Tensor:
        has_lanes = x.ndim >= 3 and x.shape[-2] == self.num_lanes
        lanes = x if has_lanes else self.expand(x)
        for layer in self.layers:
            lanes = layer(lanes, edge_index, edge_attr=edge_attr, return_lanes=True)
        if return_lanes or not self.reduce:
            return lanes
        return self.reducer(lanes)

    def mixing_stats(self) -> list[LaneMixingStats]:
        return [cast(GraphHyperConnection, layer).mixing_stats() for layer in self.layers]


class MoESequential(nn.Module):
    """Residual stack of sparse MoE layers with optional stats aggregation."""

    def __init__(
        self,
        *layers: SparseMoE,
        branch_scale: float | None = None,
        return_stats: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.return_stats = return_stats
        self.branch_scale = (
            residual_branch_scale(len(layers)) if branch_scale is None else branch_scale
        )

    def forward(
        self,
        x: Tensor,
        *,
        return_stats: bool | None = None,
    ) -> Tensor | tuple[Tensor, list[RoutingStats]]:
        should_return_stats = self.return_stats if return_stats is None else return_stats
        stats: list[RoutingStats] = []
        out = x
        for layer in self.layers:
            if should_return_stats:
                result = layer(out, return_stats=True)
                branch, layer_stats = cast(tuple[Tensor, RoutingStats], result)
                stats.append(layer_stats)
            else:
                branch = cast(Tensor, layer(out, return_stats=False))
            if branch.shape != out.shape:
                raise ValueError(
                    "MoESequential requires each MoE layer to preserve shape, "
                    f"got {tuple(branch.shape)} for branch and {tuple(out.shape)} for input"
                )
            out = out + self.branch_scale * branch
        if should_return_stats:
            return out, stats
        return out


__all__ = [
    "GraphHyperSequential",
    "HyperSequential",
    "LaneInit",
    "MoESequential",
    "residual_branch_scale",
]
