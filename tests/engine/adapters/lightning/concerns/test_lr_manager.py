"""Tests for ConfigLearningRateManager."""

from unittest.mock import MagicMock

import pytest
import torch
import torch.optim as toptim

from dlkit.engine.adapters.lightning.concerns.lr_manager import (
    ConfigLearningRateManager,
)


@pytest.fixture
def optimizer_at_lr():
    """Create an optimizer with a known learning rate."""
    param = torch.nn.Parameter(torch.tensor(1.0))
    return toptim.Adam([param], lr=3e-4)


@pytest.fixture
def module_with_configured_lr():
    """Create a mock LightningModule with configured learning rate."""
    module = MagicMock()
    module.optimizer = MagicMock()
    module.optimizer.lr = 1e-3
    module._trainer = None
    module._fabric = None
    module.hparams = {}
    return module


@pytest.fixture
def module_without_configured_lr():
    """Create a mock LightningModule without configured learning rate."""
    module = MagicMock()
    module.optimizer = MagicMock()
    module.optimizer.default_optimizer = MagicMock()
    module.optimizer.default_optimizer.lr = None
    module._trainer = None
    module._fabric = None
    module.hparams = {}
    return module


def test_get_lr_returns_configured_float(module_with_configured_lr):
    """Test that get_lr returns the configured learning rate."""
    manager = ConfigLearningRateManager(module_with_configured_lr)
    assert manager.get_lr() == pytest.approx(1e-3)


def test_get_lr_falls_back_to_trainer_optimizer(module_without_configured_lr, optimizer_at_lr):
    """Test that get_lr falls back to trainer optimizers when config LR is None."""
    trainer = MagicMock()
    trainer.optimizers = [optimizer_at_lr]
    module_without_configured_lr._trainer = trainer
    manager = ConfigLearningRateManager(module_without_configured_lr)
    assert manager.get_lr() == pytest.approx(3e-4)


def test_get_lr_returns_none_when_no_trainer(module_without_configured_lr):
    """Test that get_lr returns None when no trainer is attached."""
    manager = ConfigLearningRateManager(module_without_configured_lr)
    assert manager.get_lr() is None


def test_get_lr_returns_none_on_empty_optimizer_list(module_without_configured_lr):
    """Test that get_lr returns None when trainer has no optimizers."""
    trainer = MagicMock()
    trainer.optimizers = []
    module_without_configured_lr._trainer = trainer
    manager = ConfigLearningRateManager(module_without_configured_lr)
    assert manager.get_lr() is None


def test_set_lr_updates_trainer_param_groups(module_without_configured_lr, optimizer_at_lr):
    """Test that set_lr updates param groups in trainer optimizers."""
    from unittest.mock import patch as mock_patch

    trainer = MagicMock()
    trainer.optimizers = [optimizer_at_lr]
    module_without_configured_lr._trainer = trainer

    manager = ConfigLearningRateManager(module_without_configured_lr)
    # Mock update_settings to avoid Pydantic validation on MagicMock
    with mock_patch("dlkit.infrastructure.config.core.updater.update_settings") as mock_update:
        mock_update.return_value = module_without_configured_lr.optimizer
        manager.set_lr(5e-4)

    # Verify that the trainer optimizer param groups were updated
    assert optimizer_at_lr.param_groups[0]["lr"] == pytest.approx(5e-4)


def test_get_lr_handles_malformed_param_groups(module_without_configured_lr):
    """Test that get_lr handles errors accessing param_groups gracefully."""
    trainer = MagicMock()
    bad_optimizer = MagicMock()
    bad_optimizer.param_groups = None  # This will trigger a TypeError or KeyError
    trainer.optimizers = [bad_optimizer]
    module_without_configured_lr._trainer = trainer
    manager = ConfigLearningRateManager(module_without_configured_lr)
    # Should handle the exception and return None
    assert manager.get_lr() is None
