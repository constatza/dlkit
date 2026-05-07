"""Tests for optimizer component settings."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dlkit.infrastructure.config.optimization_selector import ParameterSelectorSettings
from dlkit.infrastructure.config.optimizer_component import (
    AdamSettings,
    AdamWSettings,
    BatchedMuonSettings,
    ConcurrentOptimizerSettings,
    MuonSettings,
)


class TestMuonSettingsFields:
    """Tests for MuonSettings field validation and structure."""

    def test_muon_settings_has_no_adamw_fields(self) -> None:
        """MuonSettings must not carry adamw_* fields — use a separate AdamW stage instead."""
        settings = MuonSettings()
        adamw_fields = [n for n in type(settings).model_fields if n.startswith("adamw_")]
        assert adamw_fields == [], (
            f"Found adamw_* fields: {adamw_fields}. "
            "Use OptimizationStageSettings(optimizer=AdamWSettings(...), "
            "selector=NonMuonSelectorSettings()) instead."
        )

    def test_muon_settings_core_hyperparams(self) -> None:
        """Verify MuonSettings has correct core hyperparameters.

        Args:
            None

        Returns:
            None
        """
        s = MuonSettings()
        assert s.lr == pytest.approx(0.02)
        assert s.momentum == pytest.approx(0.95)
        assert s.nesterov is True
        assert s.ns_steps == 5
        assert s.adjust_lr_fn == "match_rms_adamw"

    def test_batched_muon_settings_has_no_adamw_fields(self) -> None:
        """BatchedMuonSettings must stay Muon-only; companion AdamW config lives elsewhere."""
        settings = BatchedMuonSettings()
        adamw_fields = [n for n in type(settings).model_fields if n.startswith("adamw_")]
        assert adamw_fields == []

    def test_muon_settings_name_and_module(self) -> None:
        """Verify MuonSettings has correct name and module_path.

        Args:
            None

        Returns:
            None
        """
        s = MuonSettings()
        assert s.name == "Muon"
        assert s.module_path == "torch.optim"

    def test_muon_settings_immutable(self) -> None:
        """Verify MuonSettings is frozen (immutable).

        Args:
            None

        Returns:
            None
        """
        s = MuonSettings()
        with pytest.raises(Exception):  # Pydantic raises ValidationError or similar
            s.lr = 0.01


class TestConcurrentOptimizerSettingsValidation:
    """Tests for ConcurrentOptimizerSettings validator constraints."""

    def test_muon_with_other_no_selectors_is_valid(self) -> None:
        """Empty selectors are valid when at least one sub-optimizer is MuonSettings.

        Returns:
            None
        """
        settings = ConcurrentOptimizerSettings(optimizers=(MuonSettings(), AdamWSettings()))
        assert settings.selectors == ()

    def test_batched_muon_with_other_no_selectors_is_valid(self) -> None:
        """Empty selectors are valid when at least one sub-optimizer is BatchedMuonSettings."""
        settings = ConcurrentOptimizerSettings(optimizers=(BatchedMuonSettings(), AdamWSettings()))
        assert settings.selectors == ()

    def test_explicit_selectors_without_muon_is_valid(self) -> None:
        """Explicit selectors of matching length are valid for any optimizer combination.

        Returns:
            None
        """
        settings = ConcurrentOptimizerSettings(
            optimizers=(AdamWSettings(), AdamSettings()),
            selectors=(
                ParameterSelectorSettings(prefix="encoder"),
                ParameterSelectorSettings(prefix="decoder"),
            ),
        )
        assert len(settings.selectors) == 2  # noqa: PLR2004

    def test_no_muon_no_selectors_raises(self) -> None:
        """Empty selectors without any MuonSettings must raise ValidationError.

        Returns:
            None
        """
        with pytest.raises(ValidationError, match="selectors"):
            ConcurrentOptimizerSettings(optimizers=(AdamWSettings(), AdamSettings()))

    def test_mismatched_selector_length_raises(self) -> None:
        """Selectors length differing from optimizers length must raise ValidationError.

        Returns:
            None
        """
        with pytest.raises(ValidationError, match="selectors length"):
            ConcurrentOptimizerSettings(
                optimizers=(MuonSettings(), AdamWSettings()),
                selectors=(ParameterSelectorSettings(prefix="encoder"),),
            )

    def test_two_muon_no_selectors_raises(self) -> None:
        """Two Muon-family optimizers with empty selectors must raise ValidationError.

        Auto-selector inference assigns MuonEligibleSelector to every Muon-family
        optimizer, producing duplicate parameter groups when there is more than one.
        """
        with pytest.raises(ValidationError, match="exactly one Muon-family"):
            ConcurrentOptimizerSettings(optimizers=(MuonSettings(), BatchedMuonSettings()))
