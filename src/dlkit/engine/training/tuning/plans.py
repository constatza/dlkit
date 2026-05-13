"""Compatibility planning for Lightning LR tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from dlkit.infrastructure.config.optimizer_component import (
    ConcurrentOptimizerSettings,
    OptimizerSpec,
    optimizer_requires_closure,
)
from dlkit.infrastructure.config.optimizer_policy import OptimizerPolicySettings


@runtime_checkable
class ILRTunable(Protocol):
    """Protocol for models whose learning rate can be updated in-place."""

    @property
    def lr(self) -> float | None: ...

    @lr.setter
    def lr(self, value: float) -> None: ...


@dataclass(frozen=True, slots=True)
class SupportedLRTuningPlan:
    """LR-tuning plan for policies that can be projected into Lightning LR finder."""

    projected_policy: OptimizerPolicySettings

    def apply_suggested_lr(self, model: ILRTunable, lr: float) -> None:
        """Apply the suggested LR to stage-0 of the real training model."""
        model.lr = lr


@dataclass(frozen=True, slots=True)
class UnsupportedLRTuningPlan:
    """LR-tuning plan for policies Lightning LR finder cannot safely handle."""

    reason: str


type LRTuningPlan = SupportedLRTuningPlan | UnsupportedLRTuningPlan


class FirstStageTuningPolicyAdapter:
    """Project a real optimizer policy into a single-stage tuning policy."""

    def adapt(self, policy: OptimizerPolicySettings) -> OptimizerPolicySettings:
        """Return a tuning-only projection of the optimizer policy."""
        if not policy.stages:
            return policy.model_copy(deep=True)
        projected_stage = policy.stages[0].model_copy(deep=True, update={"trigger": None})
        return OptimizerPolicySettings(stages=(projected_stage,))


class LRTuningPlanBuilder:
    """Build a compatibility plan for Lightning LR tuning."""

    def __init__(self, adapter: FirstStageTuningPolicyAdapter | None = None) -> None:
        self._adapter = adapter or FirstStageTuningPolicyAdapter()

    def build(self, policy: OptimizerPolicySettings) -> LRTuningPlan:
        """Build a supported or unsupported plan for the given optimizer policy."""
        optimizer = _resolve_tuning_optimizer(policy)
        if isinstance(optimizer, ConcurrentOptimizerSettings):
            return UnsupportedLRTuningPlan(
                "Lightning LR finder does not support concurrent optimizers for stage-0 tuning."
            )
        if optimizer_requires_closure(optimizer):
            return UnsupportedLRTuningPlan(
                f"{type(optimizer).__name__.removesuffix('Settings')} requires closure-based "
                "stepping, which Lightning LR finder does not support."
            )
        projected_policy = self._adapter.adapt(policy)
        return SupportedLRTuningPlan(projected_policy=projected_policy)


def get_lr_tuning_plan(policy: OptimizerPolicySettings) -> LRTuningPlan:
    """Build an LR-tuning plan for the given optimizer policy."""
    return LRTuningPlanBuilder().build(policy)


def _resolve_tuning_optimizer(policy: OptimizerPolicySettings) -> OptimizerSpec:
    """Return the optimizer spec that Lightning LR finder would tune."""
    if not policy.stages:
        return policy.default_optimizer
    return policy.stages[0].optimizer
