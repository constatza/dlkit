"""Tests for optimizer component settings."""

from __future__ import annotations

import pytest

from dlkit.infrastructure.config.optimizer_component import MuonSettings


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
