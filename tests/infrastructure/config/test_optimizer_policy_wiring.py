"""Tests for strict optimizer-policy wiring through wrapper and training settings."""

from __future__ import annotations

import pytest

from dlkit.infrastructure.config.model_components import WrapperComponentSettings
from dlkit.infrastructure.config.optimizer_component import (
    AdamSettings,
    AdamWSettings,
    MuonSettings,
    ReduceLROnPlateauSettings,
)
from dlkit.infrastructure.config.optimizer_policy import OptimizerPolicySettings


def _build_policy(settings: WrapperComponentSettings) -> OptimizerPolicySettings:
    from dlkit.engine.workflows.factories.component_builders import build_wrapper_components

    components = build_wrapper_components(settings, entry_configs=())
    return components.optimizer_policy_settings


class TestWrapperOptimizerPolicyWiring:
    """Wrapper settings should forward policy objects without legacy migration."""

    def test_default_wrapper_uses_default_policy_shape(self) -> None:
        settings = WrapperComponentSettings()

        assert isinstance(settings.optimizer, OptimizerPolicySettings)
        assert isinstance(settings.optimizer.default_optimizer, AdamWSettings)
        assert settings.optimizer.default_scheduler is None

    def test_muon_policy_round_trips_without_field_loss(self) -> None:
        policy = OptimizerPolicySettings(
            default_optimizer=MuonSettings(lr=0.02, momentum=0.85, ns_steps=3),
        )
        built = _build_policy(WrapperComponentSettings(optimizer=policy))

        assert isinstance(built.default_optimizer, MuonSettings)
        assert built.default_optimizer.lr == pytest.approx(0.02)
        assert built.default_optimizer.momentum == pytest.approx(0.85)
        assert built.default_optimizer.ns_steps == 3

    def test_scheduler_policy_round_trips_without_implicit_scheduler(self) -> None:
        policy = OptimizerPolicySettings(
            default_scheduler=ReduceLROnPlateauSettings(
                mode="max",
                threshold=1e-3,
                patience=5,
            )
        )
        built = _build_policy(WrapperComponentSettings(optimizer=policy))

        assert isinstance(built.default_scheduler, ReduceLROnPlateauSettings)
        assert built.default_scheduler.mode == "max"
        assert built.default_scheduler.threshold == pytest.approx(1e-3)
        assert built.default_scheduler.patience == 5


class TestTrainingSettingsOptimizerPolicyField:
    """Training settings should expose optimizer policy as the only optimizer entry point."""

    def test_training_settings_default_optimizer_is_policy(self) -> None:
        from dlkit.infrastructure.config.training_settings import TrainingSettings

        settings = TrainingSettings()

        assert isinstance(settings.optimizer, OptimizerPolicySettings)
        assert isinstance(settings.optimizer.default_optimizer, AdamWSettings)
        assert settings.optimizer.default_scheduler is None

    def test_training_settings_accepts_optimizer_policy(self) -> None:
        from dlkit.infrastructure.config.training_settings import TrainingSettings

        policy = OptimizerPolicySettings(default_optimizer=AdamWSettings(lr=5e-4))
        settings = TrainingSettings(optimizer=policy)

        assert isinstance(settings.optimizer.default_optimizer, AdamWSettings | AdamSettings)
        assert settings.optimizer.default_optimizer.lr == pytest.approx(5e-4)
