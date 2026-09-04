"""Tests for TrainerSettings, including the overfit_batches capacity sanity check.

Lightning 2.6's ``Trainer(overfit_batches=...)`` (see
``lightning.pytorch.trainer.setup._init_debugging_flags``) sets
``limit_train_batches`` and ``limit_val_batches`` to the same value, each
resolved independently against its own dataloader with shuffling disabled
(``_resolve_overfit_batches``). Concretely:

- Train and val get the same *count* limit N (``limit_train_batches ==
  limit_val_batches == overfit_batches``), applied separately to each
  dataloader - this is a shared number of batches, not shared samples.
  Each of the N batches then repeats identically every epoch (no fresh
  sampling per epoch).
- Train and val do *not* share data: val is truncated on its own dataloader,
  not swapped for the train batch, so the samples themselves differ between
  the two even though both are limited to N. (Older Lightning versions
  reused the train set for val/test; this build does not.)
- The test split is untouched - overfit_batches never sets
  limit_test_batches/limit_predict_batches. Use limit_test_batches
  separately if you also want a restricted test run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn

from dlkit.infrastructure.config.compute_settings import (
    KubeflowComputeSettings,
    SlurmComputeSettings,
)
from dlkit.infrastructure.config.core.context import BuildContext
from dlkit.infrastructure.config.core.factories import FactoryProvider
from dlkit.infrastructure.config.run_settings import RunSettings
from dlkit.infrastructure.config.trainer_settings import (
    CallbackSettings,
    LoggerSettings,
    TrainerSettings,
)


class _RecordingProbe(LightningModule):
    """Records the exact x value of every batch it's shown, per stage."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.seen_train_x: list[float] = []
        self.seen_val_x: list[float] = []
        self.seen_test_x: list[float] = []

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        self.seen_train_x.append(x.item())
        return nn.functional.mse_loss(self.linear(x), y)

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, _y = batch
        self.seen_val_x.append(x.item())

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, _y = batch
        self.seen_test_x.append(x.item())

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=0.1)


def test_trainer_settings_build_creates_callbacks_and_logger(monkeypatch):
    # Prepare settings with one callback and a logger
    ts = TrainerSettings(
        callbacks=(CallbackSettings(name="ModelSummary"),),
        logger=LoggerSettings(name="CSVLogger"),
        max_epochs=1,
    )

    # Patch FactoryProvider.create_component so that callbacks/loggers resolve to simple objects
    def _fake_create(settings, ctx: BuildContext):
        # Return a simple object representing constructed component
        class _Dummy:
            pass

        return _Dummy()

    monkeypatch.setattr(FactoryProvider, "create_component", staticmethod(_fake_create))

    trainer = ts.build()
    # Trainer is constructed via FactoryProvider (our dummy class), but we verify no exceptions and attrs used
    assert trainer is not None


def test_trainer_settings_build_disables_model_summary_and_respects_progress_bar(
    monkeypatch,
):
    captured_overrides: dict | None = None

    def _fake_create(settings, ctx: BuildContext):
        nonlocal captured_overrides
        captured_overrides = ctx.overrides

        class _Dummy:
            pass

        return _Dummy()

    monkeypatch.setattr(FactoryProvider, "create_component", staticmethod(_fake_create))

    with patch(
        "dlkit.infrastructure.config.trainer_settings.should_enable_progress_bar",
        return_value=False,
    ):
        trainer = TrainerSettings().build()

    assert trainer is not None
    assert captured_overrides is not None
    assert captured_overrides["enable_model_summary"] is False
    assert captured_overrides["enable_progress_bar"] is False


def test_trainer_settings_build_applies_overfit_batches_to_real_trainer():
    # No FactoryProvider mocking: exercises the real settings -> kwargs-filtering
    # -> Trainer construction path, so a signature/filtering regression would fail here.
    trainer = TrainerSettings(overfit_batches=1, max_epochs=1, accelerator="cpu").build()
    # `overfit_batches` is set dynamically by Lightning's
    # `setup._init_debugging_flags` outside `Trainer.__init__`, so it is a
    # genuine runtime attribute invisible to static attribute inference.
    assert cast(Any, trainer).overfit_batches == 1


def test_overfit_batches_repeats_the_identical_sample_every_epoch(train_val_test_dataloaders):
    """overfit_batches=1 must reuse each split's own sample every epoch.

    Not just "few batches" - the train loop repeats its one train sample and
    the val loop repeats its one (different) val sample, since a capacity
    check depends on the model seeing identical inputs each pass.
    """
    train_loader, val_loader, _test_loader = train_val_test_dataloaders
    # `num_sanity_val_steps` is not a declared TrainerSettings field - it is
    # accepted only via the model's `extra="allow"` config and forwarded to
    # Trainer by the factory's kwargs filtering. `model_validate` exercises
    # that same runtime path without a statically-unknown keyword argument.
    trainer = TrainerSettings.model_validate(
        {
            "overfit_batches": 1,
            "max_epochs": 5,
            "accelerator": "cpu",
            "enable_checkpointing": False,
            "num_sanity_val_steps": 0,
        }
    ).build()
    model = _RecordingProbe()

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Only 1 of the loader's 20/5 batches was used per epoch...
    assert trainer.num_training_batches == 1
    assert trainer.num_val_batches[0] == 1
    # ...and it was the *same* one every epoch, not a fresh sample each time.
    assert len(model.seen_train_x) == 5
    assert len(set(model.seen_train_x)) == 1
    assert len(model.seen_val_x) == 5
    assert len(set(model.seen_val_x)) == 1


def test_overfit_batches_does_not_alias_val_to_the_train_sample(train_val_test_dataloaders):
    """Train and val are restricted independently, not aliased to each other.

    Easy to assume (from older Lightning docs/versions) that overfit_batches
    makes validation reuse the train batch. In this Lightning version it
    doesn't: val is truncated on its own dataloader.
    """
    train_loader, val_loader, _test_loader = train_val_test_dataloaders
    trainer = TrainerSettings.model_validate(
        {
            "overfit_batches": 1,
            "max_epochs": 1,
            "accelerator": "cpu",
            "enable_checkpointing": False,
            "num_sanity_val_steps": 0,
        }
    ).build()
    model = _RecordingProbe()

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Lightning restricts train and val to 1 batch *each*, independently -
    # it does not swap in the train batch for validation.
    assert model.seen_train_x[0] != model.seen_val_x[0]


class _LossLoggingProbe(LightningModule):
    """Minimal model that logs a real val_loss, for ModelCheckpoint(monitor=...)."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        return nn.functional.mse_loss(self.linear(x), y)

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        self.log("val_loss", nn.functional.mse_loss(self.linear(x), y))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=0.1)


def test_default_style_checkpoint_bounds_files_to_one(tmp_path, train_val_test_dataloaders):
    """The smart-default ModelCheckpoint shape (save_top_k=1, step/epoch-free
    filename, best-only) must produce at most 1 checkpoint file - not one per
    epoch - regardless of how many epochs run.

    Uses overfit_batches=1 + a tiny dataset to keep this cheap.
    """
    train_loader, val_loader, _test_loader = train_val_test_dataloaders
    checkpoint_cb = ModelCheckpoint(
        dirpath=tmp_path,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best",
    )
    # Construct the real Trainer directly (not via TrainerSettings.build()) so
    # the checkpoint_cb passed in callbacks= is the only ModelCheckpoint Lightning
    # sees - build() would otherwise leave enable_checkpointing=True with an
    # empty explicit callback list, causing Lightning to auto-inject its own
    # second, unconfined default ModelCheckpoint (dirpath=None -> CWD).
    trainer = Trainer(
        overfit_batches=1,
        max_epochs=5,
        accelerator="cpu",
        enable_checkpointing=True,
        callbacks=[checkpoint_cb],
        default_root_dir=tmp_path,
        logger=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    model = _LossLoggingProbe()

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    ckpt_files = list(Path(tmp_path).glob("*.ckpt"))
    assert len(ckpt_files) <= 1


def test_devices_and_num_nodes_mirror_real_trainer_constructor_params():
    """TrainerSettings.devices/num_nodes must stay named/shaped to match
    Trainer.__init__ exactly, the same way accelerator/strategy/precision
    already do — that's what lets them flow through the standard
    settings-to-kwargs reflection pipeline with no special-casing."""
    import inspect

    params = inspect.signature(Trainer.__init__).parameters
    assert "devices" in params
    assert "num_nodes" in params


def test_build_resolves_devices_and_num_nodes_from_compute_environment(monkeypatch):
    """Unset compute.devices/num_nodes fall back to the resolved environment
    (SLURM here) rather than Lightning's own defaults."""
    monkeypatch.setenv("SLURM_NTASKS", "8")
    monkeypatch.setenv("SLURM_NNODES", "2")
    monkeypatch.setenv("SLURM_NTASKS_PER_NODE", "4")
    captured_overrides: dict | None = None

    def _fake_create(settings, ctx: BuildContext):
        nonlocal captured_overrides
        captured_overrides = ctx.overrides

        class _Dummy:
            pass

        return _Dummy()

    monkeypatch.setattr(FactoryProvider, "create_component", staticmethod(_fake_create))

    TrainerSettings().build()

    assert captured_overrides is not None
    assert captured_overrides["devices"] == 4
    assert captured_overrides["num_nodes"] == 2


def test_build_explicit_trainer_devices_are_never_overridden_by_resolved_topology(monkeypatch):
    """TrainerSettings.devices/num_nodes (mirroring Trainer's own constructor)
    always win over the environment's resolved topology, even when SLURM env
    vars suggest different values."""
    monkeypatch.setenv("SLURM_NTASKS", "8")
    monkeypatch.setenv("SLURM_NNODES", "2")
    monkeypatch.setenv("SLURM_NTASKS_PER_NODE", "4")
    captured_overrides: dict | None = None

    def _fake_create(settings, ctx: BuildContext):
        nonlocal captured_overrides
        captured_overrides = ctx.overrides

        class _Dummy:
            pass

        return _Dummy()

    monkeypatch.setattr(FactoryProvider, "create_component", staticmethod(_fake_create))

    session = RunSettings(compute=SlurmComputeSettings())
    TrainerSettings(devices=1, num_nodes=1).build(session=session)

    assert captured_overrides is not None
    assert captured_overrides["devices"] == 1
    assert captured_overrides["num_nodes"] == 1


def test_build_omits_devices_and_num_nodes_when_environment_cannot_derive_them(monkeypatch):
    """Under 'auto', a detected environment that can't derive topology (LSF
    here) must leave devices/num_nodes out of overrides entirely so
    Lightning's own defaults apply, rather than passing an explicit None that
    could confuse Trainer construction. (Explicitly selecting `environment =
    "lsf"` in config requires devices/num_nodes up front — see
    test_compute_settings.py — so this scenario is reached via 'auto'
    detecting LSF, not via an explicit LSFComputeSettings.)"""
    monkeypatch.setenv("LSB_JOBID", "12345")
    monkeypatch.setenv("LSB_DJOB_RANKFILE", "/tmp/dlkit-test-rankfile-does-not-need-to-exist")
    monkeypatch.setenv("JSM_NAMESPACE_LOCAL_RANK", "0")
    monkeypatch.setenv("JSM_NAMESPACE_SIZE", "4")
    captured_overrides: dict | None = None

    def _fake_create(settings, ctx: BuildContext):
        nonlocal captured_overrides
        captured_overrides = ctx.overrides

        class _Dummy:
            pass

        return _Dummy()

    monkeypatch.setattr(FactoryProvider, "create_component", staticmethod(_fake_create))

    TrainerSettings().build()  # default session=None -> AutoComputeSettings()

    assert captured_overrides is not None
    assert "devices" not in captured_overrides
    assert "num_nodes" not in captured_overrides
    assert "plugins" not in captured_overrides


def test_build_forwards_kubeflow_cluster_environment_as_a_plugin(monkeypatch):
    from lightning.fabric.plugins.environments import KubeflowEnvironment

    captured_overrides: dict | None = None

    def _fake_create(settings, ctx: BuildContext):
        nonlocal captured_overrides
        captured_overrides = ctx.overrides

        class _Dummy:
            pass

        return _Dummy()

    monkeypatch.setattr(FactoryProvider, "create_component", staticmethod(_fake_create))

    session = RunSettings(compute=KubeflowComputeSettings(devices=1, num_nodes=1))
    TrainerSettings().build(session=session)

    assert captured_overrides is not None
    assert isinstance(captured_overrides["plugins"][0], KubeflowEnvironment)


def test_build_uses_default_compute_settings_when_no_session_is_given():
    """Without a session, compute topology resolves against defaults (auto ->
    local, since no scheduler env vars are set), not an error."""
    trainer = TrainerSettings(accelerator="cpu").build(session=None)

    assert trainer.num_nodes == 1


def test_build_passes_explicit_strategy_straight_through_to_real_trainer():
    """strategy is a plain Lightning passthrough field — DLKit does not
    select or validate it, it just has to reach the real Trainer unchanged."""
    trainer = TrainerSettings(accelerator="cpu", strategy="ddp", devices=1, num_nodes=1).build(
        session=None
    )

    assert type(trainer.strategy).__name__ == "DDPStrategy"


def test_overfit_batches_leaves_the_test_split_unrestricted(train_val_test_dataloaders):
    """overfit_batches only sets limit_train_batches/limit_val_batches.

    limit_test_batches is untouched, so trainer.test() still runs the full
    test split even with overfit_batches configured on the same Trainer.
    """
    _train_loader, _val_loader, test_loader = train_val_test_dataloaders
    trainer = TrainerSettings(
        overfit_batches=1,
        accelerator="cpu",
        enable_checkpointing=False,
    ).build()
    model = _RecordingProbe()

    trainer.test(model, dataloaders=test_loader)

    # overfit_batches only overrides limit_train_batches/limit_val_batches;
    # the full test split still runs.
    assert trainer.num_test_batches[0] == 5
    assert len(model.seen_test_x) == 5
