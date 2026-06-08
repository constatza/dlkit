"""Tests for LR-tuning compatibility planning."""

from __future__ import annotations

from unittest.mock import Mock

from dlkit.engine.training.tuning import (
    SupportedLRTuningPlan,
    UnsupportedLRTuningPlan,
    get_lr_tuning_plan,
)
from dlkit.infrastructure.config.optimization_selector import ParameterSelectorSettings
from dlkit.infrastructure.config.optimization_stage import OptimizationStageSettings
from dlkit.infrastructure.config.optimization_trigger import TriggerSettings
from dlkit.infrastructure.config.optimizer_component import (
    AdamSettings,
    AdamWSettings,
    ConcurrentOptimizerSettings,
    LBFGSSettings,
    StepLRSettings,
)
from dlkit.infrastructure.config.optimizer_policy import OptimizerPolicySettings


def test_single_stage_policy_is_supported() -> None:
    """Single-stage policies should use Lightning LR finder directly."""
    policy = OptimizerPolicySettings(default_optimizer=AdamWSettings(lr=1e-3))

    plan = get_lr_tuning_plan(policy)

    assert isinstance(plan, SupportedLRTuningPlan)
    assert plan.projected_policy.default_optimizer == policy.default_optimizer
    assert plan.projected_policy.default_scheduler is None


def test_multi_stage_policy_projects_only_first_stage() -> None:
    """Sequential staged policies should project only stage 0 for LR tuning."""
    policy = OptimizerPolicySettings(
        stages=(
            OptimizationStageSettings(
                optimizer=AdamWSettings(lr=1e-3),
                scheduler=StepLRSettings(step_size=1, gamma=0.5),
                trigger=TriggerSettings(at_epoch=5),
            ),
            OptimizationStageSettings(
                optimizer=AdamSettings(lr=2e-3),
                scheduler=StepLRSettings(step_size=1, gamma=0.1),
            ),
        )
    )

    plan = get_lr_tuning_plan(policy)

    assert isinstance(plan, SupportedLRTuningPlan)
    assert len(plan.projected_policy.stages) == 1
    assert plan.projected_policy.stages[0].optimizer == policy.stages[0].optimizer
    assert plan.projected_policy.stages[0].scheduler is None
    assert plan.projected_policy.stages[0].trigger is None


def test_apply_suggested_lr_targets_first_stage_only() -> None:
    """Applying the suggested LR should only update the first active stage."""
    model = Mock()
    model.lr = 1e-3
    policy = OptimizerPolicySettings(
        stages=(
            OptimizationStageSettings(
                optimizer=AdamWSettings(lr=1e-3),
                trigger=TriggerSettings(at_epoch=5),
            ),
            OptimizationStageSettings(optimizer=AdamSettings(lr=2e-3)),
        )
    )
    plan = get_lr_tuning_plan(policy)
    assert isinstance(plan, SupportedLRTuningPlan)

    plan.apply_suggested_lr(model, 0.02)

    assert model.lr == 0.02


def test_concurrent_first_stage_is_unsupported() -> None:
    """Concurrent first-stage policies are rejected explicitly in v1."""
    policy = OptimizerPolicySettings(
        default_optimizer=ConcurrentOptimizerSettings(
            optimizers=(AdamWSettings(lr=1e-3), AdamSettings(lr=1e-3)),
            selectors=(
                ParameterSelectorSettings(prefix="network.0"),
                ParameterSelectorSettings(prefix="network.2"),
            ),
        )
    )

    plan = get_lr_tuning_plan(policy)

    assert isinstance(plan, UnsupportedLRTuningPlan)
    assert "concurrent optimizers" in plan.reason


def test_lbfgs_first_stage_is_unsupported_for_lr_finder() -> None:
    """LBFGS remains unsupported for LR finder even though schedulers are allowed."""
    policy = OptimizerPolicySettings(default_optimizer=LBFGSSettings())

    plan = get_lr_tuning_plan(policy)

    assert isinstance(plan, UnsupportedLRTuningPlan)
    assert "closure-based" in plan.reason


def test_ilr_tunable_exported_from_tuning_package() -> None:
    """ILRTunable must be importable from the tuning package, not vanilla_executor."""
    from dlkit.engine.training.tuning import ILRTunable

    assert callable(ILRTunable)


def test_supported_plan_has_no_target_stage_index() -> None:
    """SupportedLRTuningPlan must not expose target_stage_index — it was always 0."""
    from dlkit.infrastructure.config.optimizer_policy import OptimizerPolicySettings

    plan = SupportedLRTuningPlan(projected_policy=OptimizerPolicySettings())
    assert not hasattr(plan, "target_stage_index")


def test_projected_stage_strips_trigger() -> None:
    """Projected single-stage policy must not carry the original trigger."""
    policy = OptimizerPolicySettings(
        stages=(
            OptimizationStageSettings(
                optimizer=AdamWSettings(lr=1e-3),
                scheduler=StepLRSettings(step_size=1, gamma=0.5),
                trigger=TriggerSettings(at_epoch=5),
            ),
            OptimizationStageSettings(optimizer=AdamSettings(lr=2e-3)),
        )
    )
    plan = get_lr_tuning_plan(policy)
    assert isinstance(plan, SupportedLRTuningPlan)
    assert plan.projected_policy.stages[0].trigger is None
