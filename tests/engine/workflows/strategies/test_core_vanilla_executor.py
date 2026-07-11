"""Tests for VanillaExecutor following SOLID principles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dlkit.common import TrainingResult, WorkflowError
from dlkit.engine.training import VanillaExecutor
from dlkit.engine.training._executor_helpers import _reload_best_checkpoint_weights
from dlkit.engine.training.components import RuntimeComponents
from dlkit.infrastructure.config.job_config import TrainingJobConfig


@pytest.fixture
def trainer_stub():
    """Create a trainer stub for testing."""

    class TrainerStub:
        def __init__(self):
            self.called = {"fit": 0, "predict": 0, "test": 0, "validate": 0}
            self.logged_metrics = {"val_loss": 1.23, "acc": 0.9, "tensor_val": "non_numeric"}
            self.callbacks = []
            self.checkpoint_callback = None

        def fit(self, *args, **kwargs):
            self.called["fit"] += 1

        def validate(self, *args, **kwargs):
            self.called["validate"] += 1

        def predict(self, *args, **kwargs):
            self.called["predict"] += 1

        def test(self, *args, **kwargs):
            self.called["test"] += 1

    return TrainerStub()


@pytest.fixture
def build_components(trainer_stub):
    """Create RuntimeComponents for testing."""

    @dataclass(frozen=True, slots=True)
    class DummyModel:
        pass

    return RuntimeComponents(
        model=cast(Any, DummyModel()),
        datamodule=cast(Any, object()),
        trainer=cast(Any, trainer_stub),
        meta={},
    )


@pytest.fixture
def settings() -> TrainingJobConfig:
    """Create basic settings for testing.

    No data entries are needed: VanillaExecutor only accesses ``run.seed``
    and ``training.*`` fields; dataset paths are irrelevant here.
    """
    return TrainingJobConfig.model_validate(
        {
            "run": {"type": "train", "seed": 42},
            "experiment": {"name": "test"},
            "model": {"class": "Dummy", "module_path": "dlkit.domain.nn"},
            "data": {
                "batch_size": 8,
                "num_workers": 0,
            },
            "training": {
                "loss": "mse",
                "trainer": {"max_epochs": 2, "accelerator": "cpu"},
                "optimizer": {"name": "AdamW", "lr": 1e-3},
            },
        }
    )


def test_vanilla_executor_single_responsibility(
    build_components: RuntimeComponents, settings: TrainingJobConfig, trainer_stub
) -> None:
    """Test that VanillaExecutor has single responsibility: pure training execution."""
    executor = VanillaExecutor()

    result = executor.execute(build_components, settings)

    # Verify training execution
    assert trainer_stub.called["fit"] == 1
    assert trainer_stub.called["predict"] == 1
    assert trainer_stub.called["test"] == 1

    # Verify result structure
    assert isinstance(result, TrainingResult)
    assert isinstance(result.metrics, dict)
    assert isinstance(result.artifacts, dict)
    assert result.duration_seconds == 0.0


def test_vanilla_executor_metric_extraction(
    build_components: RuntimeComponents, settings: TrainingJobConfig, trainer_stub
) -> None:
    """Test metric extraction from trainer."""
    executor = VanillaExecutor()

    result = executor.execute(build_components, settings)

    # Numeric metrics should be converted to float
    assert result.metrics["val_loss"] == 1.23
    assert result.metrics["acc"] == 0.9

    # Non-numeric values should be preserved as-is
    assert result.metrics["tensor_val"] == "non_numeric"


def test_vanilla_executor_no_trainer_error(settings: TrainingJobConfig) -> None:
    """Test that executor raises WorkflowError when trainer is None."""
    components = RuntimeComponents(
        model=cast(Any, object()),
        datamodule=cast(Any, object()),
        trainer=None,  # No trainer
        meta={},
    )

    executor = VanillaExecutor()

    with pytest.raises(WorkflowError) as exc_info:
        executor.execute(components, settings)

    assert "Trainer is required for training" in str(exc_info.value)
    assert exc_info.value.context["stage"] == "execute"


def test_vanilla_executor_trainer_exception_handling(settings: TrainingJobConfig) -> None:
    """Test that trainer exceptions are properly wrapped."""

    class FailingTrainer:
        def fit(self, *args, **kwargs):
            raise RuntimeError("Training failed")

    components = RuntimeComponents(
        model=cast(Any, object()),
        datamodule=cast(Any, object()),
        trainer=cast(Any, FailingTrainer()),
        meta={},
    )

    executor = VanillaExecutor()

    with pytest.raises(WorkflowError) as exc_info:
        executor.execute(components, settings)

    assert "Vanilla execution failed" in str(exc_info.value.message)
    assert "Training failed" in str(exc_info.value.message)
    assert exc_info.value.context["stage"] == "execute"


def test_vanilla_executor_post_training_exception_handling(
    build_components: RuntimeComponents, settings: TrainingJobConfig
) -> None:
    """Test that predict/test exceptions don't fail the entire execution."""

    class PartiallyFailingTrainer:
        def __init__(self):
            self.called = {"fit": 0, "predict": 0, "test": 0}
            self.logged_metrics = {"val_loss": 1.5}
            self.callbacks = []

        def fit(self, *args, **kwargs):
            self.called["fit"] += 1

        def predict(self, *args, **kwargs):
            self.called["predict"] += 1
            raise RuntimeError("Predict failed")

        def test(self, *args, **kwargs):
            self.called["test"] += 1
            raise RuntimeError("Test failed")

    failing_trainer = PartiallyFailingTrainer()
    components = RuntimeComponents(
        model=cast(Any, object()),
        datamodule=cast(Any, object()),
        trainer=cast(Any, failing_trainer),
        meta={},
    )

    executor = VanillaExecutor()

    # Should not raise exception despite predict/test failures
    result = executor.execute(components, settings)

    assert isinstance(result, TrainingResult)
    assert failing_trainer.called["fit"] == 1
    assert failing_trainer.called["predict"] == 1  # Called but failed
    assert failing_trainer.called["test"] == 1  # Called but failed


def test_execute_reloads_best_checkpoint_when_available(
    build_components: RuntimeComponents, settings: TrainingJobConfig, trainer_stub
) -> None:
    """execute() must reload via ckpt_path='best' when a best checkpoint exists."""
    checkpoint_cb = ModelCheckpoint()
    checkpoint_cb.best_model_path = "/fake/best.ckpt"
    trainer_stub.checkpoint_callback = checkpoint_cb

    executor = VanillaExecutor()
    executor.execute(build_components, settings)

    assert trainer_stub.called["validate"] == 1


def test_execute_skips_reload_when_no_checkpoint_callback(
    build_components: RuntimeComponents, settings: TrainingJobConfig, trainer_stub
) -> None:
    """No ModelCheckpoint configured (e.g. enable_checkpointing=False) -> no reload."""
    executor = VanillaExecutor()
    executor.execute(build_components, settings)

    assert trainer_stub.called["validate"] == 0


def test_execute_skips_reload_when_best_model_path_is_empty(
    build_components: RuntimeComponents, settings: TrainingJobConfig, trainer_stub
) -> None:
    """A configured ModelCheckpoint with no recorded best path (e.g. validation
    never ran) must not trigger a reload."""
    checkpoint_cb = ModelCheckpoint()
    checkpoint_cb.best_model_path = ""
    trainer_stub.checkpoint_callback = checkpoint_cb

    executor = VanillaExecutor()
    executor.execute(build_components, settings)

    assert trainer_stub.called["validate"] == 0


def test_execute_raises_workflow_error_when_reload_fails(
    build_components: RuntimeComponents, settings: TrainingJobConfig, trainer_stub
) -> None:
    """A configured-but-broken checkpoint must fail loudly, not be swallowed."""
    checkpoint_cb = ModelCheckpoint()
    checkpoint_cb.best_model_path = "/fake/best.ckpt"
    trainer_stub.checkpoint_callback = checkpoint_cb

    def _broken_validate(*args, **kwargs):
        raise RuntimeError("checkpoint file is corrupt")

    trainer_stub.validate = _broken_validate

    executor = VanillaExecutor()
    with pytest.raises(WorkflowError, match="checkpoint file is corrupt"):
        executor.execute(build_components, settings)


class _TinyProbe(LightningModule):
    """Minimal real LightningModule for exercising a genuine checkpoint save/reload."""

    def __init__(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self._train_loader = train_loader
        self._val_loader = val_loader

    def train_dataloader(self) -> DataLoader:
        return self._train_loader

    def val_dataloader(self) -> DataLoader:
        return self._val_loader

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        return nn.functional.mse_loss(self.linear(x), y)

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        self.log("val_loss", nn.functional.mse_loss(self.linear(x), y))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=0.1)


def test_reload_best_checkpoint_weights_restores_real_checkpoint_via_ckpt_path_best(
    tmp_path: Path,
) -> None:
    """Direct, real-Lightning proof that ckpt_path="best" genuinely restores the
    checkpoint file's weights, not whatever the in-memory model was mutated to.

    Uses overfit_batches=1 and a 4-row dataset to keep this cheap, per the
    project's capacity-sanity-check convention (see test_trainer_settings_build.py).
    """
    torch.manual_seed(0)
    x = torch.randn(4, 1)
    y = torch.randn(4, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    model = _TinyProbe(loader, loader)

    checkpoint_cb = ModelCheckpoint(dirpath=tmp_path, monitor="val_loss", mode="min", save_top_k=1)
    trainer = Trainer(
        max_epochs=2,
        overfit_batches=1,
        accelerator="cpu",
        enable_checkpointing=True,
        callbacks=[checkpoint_cb],
        default_root_dir=tmp_path,
        logger=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )

    trainer.fit(model)
    assert checkpoint_cb.best_model_path

    saved = torch.load(checkpoint_cb.best_model_path, weights_only=False)
    saved_weight = saved["state_dict"]["linear.weight"].clone()

    # Simulate "training continued past the best epoch": corrupt the in-memory
    # weights so they no longer match the checkpoint on disk.
    with torch.no_grad():
        model.linear.weight.fill_(999.0)
    assert not torch.equal(model.linear.weight, saved_weight)

    _reload_best_checkpoint_weights(model, trainer, datamodule=None)

    assert torch.equal(model.linear.weight, saved_weight)


class _DivergingProbe(LightningModule):
    """Logs a monitored metric that gets worse every epoch, so the best
    checkpoint (epoch 0) provably diverges from the final in-memory weights
    (epoch N-1) after real gradient updates."""

    def __init__(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self._train_loader = train_loader
        self._val_loader = val_loader

    def train_dataloader(self) -> DataLoader:
        return self._train_loader

    def val_dataloader(self) -> DataLoader:
        return self._val_loader

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        return nn.functional.mse_loss(self.linear(x), y)

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        # Monotonically worse each epoch (mode="min"), independent of real loss,
        # so best_model_path deterministically pins to epoch 0.
        self.log("val_metric", float(self.current_epoch))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=0.1)


def test_execute_returns_best_model_in_training_result_not_last_epoch(
    tmp_path: Path, settings: TrainingJobConfig
) -> None:
    """The model handed back to the user via TrainingResult.model_state.model
    must be the best checkpoint, not whatever training ended on."""
    torch.manual_seed(0)
    x = torch.randn(4, 1)
    y = torch.randn(4, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    model = _DivergingProbe(loader, loader)

    checkpoint_cb = ModelCheckpoint(
        dirpath=tmp_path, monitor="val_metric", mode="min", save_top_k=1
    )
    trainer = Trainer(
        max_epochs=3,
        overfit_batches=1,
        accelerator="cpu",
        enable_checkpointing=True,
        callbacks=[checkpoint_cb],
        default_root_dir=tmp_path,
        logger=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    components = RuntimeComponents(
        model=model, datamodule=cast(Any, None), trainer=trainer, meta={}
    )

    result = VanillaExecutor().execute(components, settings)

    # Best checkpoint is epoch 0 (val_metric=0.0 is the minimum); training ran
    # 3 epochs of real gradient descent, so epoch-0 weights provably differ
    # from whatever the final in-memory weights would have been without reload.
    best_state = torch.load(checkpoint_cb.best_model_path, weights_only=False)["state_dict"]
    assert result.model_state is not None
    assert result.model_state.model is model
    assert torch.equal(model.linear.weight, best_state["linear.weight"])


def test_reload_best_checkpoint_weights_is_noop_without_checkpoint_callback(
    tmp_path: Path,
) -> None:
    """No ModelCheckpoint on the trainer -> in-memory weights are left untouched."""
    torch.manual_seed(0)
    x = torch.randn(4, 1)
    y = torch.randn(4, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    model = _TinyProbe(loader, loader)
    trainer = Trainer(
        max_epochs=1,
        overfit_batches=1,
        accelerator="cpu",
        enable_checkpointing=False,
        default_root_dir=tmp_path,
        logger=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(model)

    with torch.no_grad():
        model.linear.weight.fill_(999.0)

    _reload_best_checkpoint_weights(model, trainer, datamodule=None)

    assert torch.equal(model.linear.weight, torch.full_like(model.linear.weight, 999.0))
