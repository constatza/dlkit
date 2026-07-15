from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class LaneMixingStats:
    """Diagnostics for multi-lane residual mixing."""

    lane_entropy: Tensor
    dominant_lane_fraction: Tensor
    off_diagonal_norm: Tensor


def _identity_matrix(num_lanes: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.eye(num_lanes, device=device, dtype=dtype)


def _row_entropy(weights: Tensor) -> Tensor:
    probs = weights.abs()
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(probs.dtype).eps)
    return -(probs * probs.clamp_min(torch.finfo(probs.dtype).eps).log()).sum(dim=-1).mean()


class LaneExpand(nn.Module):
    """Expand a feature-last tensor into repeated residual lanes."""

    def __init__(self, num_lanes: int) -> None:
        super().__init__()
        if num_lanes <= 0:
            raise ValueError("num_lanes must be positive")
        self.num_lanes = num_lanes

    def forward(self, x: Tensor) -> Tensor:
        """Return ``x`` with a new lane axis before the feature axis."""
        return x.unsqueeze(-2).expand(*x.shape[:-1], self.num_lanes, x.shape[-1])


class LaneReduce(nn.Module):
    """Reduce feature-last residual lanes with learned lane weights."""

    def __init__(self, num_lanes: int, *, identity_lane: int = 0) -> None:
        super().__init__()
        if num_lanes <= 0:
            raise ValueError("num_lanes must be positive")
        if not 0 <= identity_lane < num_lanes:
            raise ValueError("identity_lane must refer to an existing lane")
        self.num_lanes = num_lanes
        self.identity_lane = identity_lane
        logits = torch.full((num_lanes,), -8.0)
        logits[identity_lane] = 8.0
        self.logits = nn.Parameter(logits)

    @property
    def weights(self) -> Tensor:
        """Softmax-normalized lane reduction weights."""
        return torch.softmax(self.logits, dim=0)

    def forward(self, x: Tensor) -> Tensor:
        """Collapse ``Tensor[..., lanes, features]`` to ``Tensor[..., features]``."""
        if x.shape[-2] != self.num_lanes:
            raise ValueError(f"LaneReduce expected {self.num_lanes} lanes, got {x.shape[-2]}")
        return torch.sum(x * self.weights.view(*([1] * (x.ndim - 2)), self.num_lanes, 1), dim=-2)


class _BaseHyperConnection(nn.Module):
    """Shared lane-mixing state for Hyper-Connection wrappers."""

    def __init__(
        self,
        module: nn.Module,
        *,
        num_lanes: int = 2,
        branch_scale: float = 1.0,
        return_lanes: bool = False,
    ) -> None:
        super().__init__()
        if num_lanes <= 0:
            raise ValueError("num_lanes must be positive")
        if branch_scale < 0.0:
            raise ValueError("branch_scale must be non-negative")
        self.module = module
        self.num_lanes = num_lanes
        self.branch_scale = branch_scale
        self.return_lanes = return_lanes
        self.expand = LaneExpand(num_lanes)
        self.reduce = LaneReduce(num_lanes)
        self.pre_delta = nn.Parameter(torch.zeros(num_lanes, num_lanes))
        self.post_delta = nn.Parameter(torch.zeros(num_lanes, num_lanes))

    def pre_mixing_matrix(self) -> Tensor:
        """Identity-biased pre-branch lane mixing matrix."""
        eye = _identity_matrix(
            self.num_lanes,
            device=self.pre_delta.device,
            dtype=self.pre_delta.dtype,
        )
        return eye + self.pre_delta / math.sqrt(self.num_lanes)

    def post_mixing_matrix(self) -> Tensor:
        """Identity-biased post-branch lane mixing matrix."""
        eye = _identity_matrix(
            self.num_lanes,
            device=self.post_delta.device,
            dtype=self.post_delta.dtype,
        )
        return eye + self.post_delta / math.sqrt(self.num_lanes)

    def mixing_stats(self) -> LaneMixingStats:
        """Return compact diagnostics for lane utilization and mixing."""
        matrix = self.post_mixing_matrix()
        diag = torch.diagonal(matrix).abs()
        total = matrix.abs().sum().clamp_min(torch.finfo(matrix.dtype).eps)
        eye = _identity_matrix(self.num_lanes, device=matrix.device, dtype=matrix.dtype)
        off_diag = matrix * (1.0 - eye)
        return LaneMixingStats(
            lane_entropy=_row_entropy(matrix),
            dominant_lane_fraction=diag.max() / total,
            off_diagonal_norm=torch.linalg.vector_norm(off_diag),
        )

    @staticmethod
    def _mix_lanes(x: Tensor, matrix: Tensor) -> Tensor:
        return torch.einsum("...lf,ml->...mf", x, matrix)


class HyperConnection(_BaseHyperConnection):
    """Multi-lane residual wrapper generalizing a standard residual connection.

    The wrapped module is applied independently per lane. Learnable pre/post
    lane-mixing matrices are initialized at identity, so a zero branch starts as
    an exact multi-lane residual identity.
    """

    def forward(self, x: Tensor, *, return_lanes: bool | None = None) -> Tensor:
        """Apply the wrapped module through identity-biased residual lanes."""
        has_lanes = x.ndim >= 3 and x.shape[-2] == self.num_lanes
        lanes = x if has_lanes else self.expand(x)
        mixed = self._mix_lanes(lanes, self.pre_mixing_matrix())
        branch = self._apply_module_per_lane(mixed)
        out_lanes = lanes + self.branch_scale * self._mix_lanes(branch, self.post_mixing_matrix())
        should_return_lanes = self.return_lanes if return_lanes is None else return_lanes
        if should_return_lanes or has_lanes:
            return out_lanes
        return self.reduce(out_lanes)

    def _apply_module_per_lane(self, lanes: Tensor) -> Tensor:
        outputs = [self.module(lanes[..., lane, :]) for lane in range(self.num_lanes)]
        return torch.stack(outputs, dim=-2)


class GraphHyperConnection(_BaseHyperConnection):
    """Hyper-Connection wrapper for graph modules with ``edge_index`` arguments."""

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        *,
        return_lanes: bool | None = None,
    ) -> Tensor:
        """Apply a graph module independently per lane, sharing graph structure."""
        has_lanes = x.ndim >= 3 and x.shape[-2] == self.num_lanes
        lanes = x if has_lanes else self.expand(x)
        mixed = self._mix_lanes(lanes, self.pre_mixing_matrix())
        branch = self._apply_graph_module_per_lane(mixed, edge_index, edge_attr)
        out_lanes = lanes + self.branch_scale * self._mix_lanes(branch, self.post_mixing_matrix())
        should_return_lanes = self.return_lanes if return_lanes is None else return_lanes
        if should_return_lanes or has_lanes:
            return out_lanes
        return self.reduce(out_lanes)

    def _apply_graph_module_per_lane(
        self,
        lanes: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None,
    ) -> Tensor:
        outputs = [
            self.module(lanes[..., lane, :], edge_index, edge_attr=edge_attr)
            for lane in range(self.num_lanes)
        ]
        return torch.stack(outputs, dim=-2)


__all__ = [
    "GraphHyperConnection",
    "HyperConnection",
    "LaneExpand",
    "LaneMixingStats",
    "LaneReduce",
]
