"""Integration tests verifying optimization controller wiring in StandardLightningWrapper.

These tests exercise the full config → program → controller → wrapper path to confirm:
- Sequential 2-stage programs are always routed to ManualOptimizationController
  (so Lightning does NOT step inactive-stage optimizers automatically).
- Concurrent 2-optimizer groups are routed to AutomaticOptimizationController
  and return both optimizer configs (with schedulers preserved).
- Scheduler regressions work through Trainer.fit() for both sequential and
  concurrent optimizer programs without relying on parameter updates.
- _requires_manual_optimization detects LBFGS correctly.
- Both optimizers in a concurrent group actually update parameters when stepped.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, cast

import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback
from tensordict import TensorDict
from torch import nn
from torch.nn import ModuleList
from torch.utils.data import DataLoader, IterableDataset

from dlkit.engine.adapters.lightning.standard import StandardLightningWrapper
from dlkit.engine.adapters.lightning.wrapper_types import WrapperComponents
from dlkit.engine.training.optimization.builder import OptimizerPolicyBuilder
from dlkit.engine.training.optimization.concurrent_optimizer import ConcurrentOptimizer
from dlkit.engine.training.optimization.controllers import (
    AutomaticOptimizationController,
    ManualOptimizationController,
    _requires_manual_optimization,
)
from dlkit.infrastructure.config import OptimizerPolicySettings
from dlkit.infrastructure.config.data_roles import DataRole
from dlkit.infrastructure.config.entry_types import ValueEntry
from dlkit.infrastructure.config.model_components import (
    ModelComponentSettings,
    WrapperComponentSettings,
)
from dlkit.infrastructure.config.optimization_selector import ParameterSelectorSettings
from dlkit.infrastructure.config.optimization_stage import OptimizationStageSettings
from dlkit.infrastructure.config.optimization_trigger import TriggerSettings
from dlkit.infrastructure.config.optimizer_component import (
    AdamSettings,
    AdamWSettings,
    ConcurrentOptimizerSettings,
    ReduceLROnPlateauSettings,
    StepLRSettings,
)

_MODULE = "tests.engine.adapters.lightning.test_standard_wrapper_optimization"


# ---------------------------------------------------------------------------
# Minimal model registered at a predictable module path
# ---------------------------------------------------------------------------


class _TwoLayer(nn.Module):
    """Two-layer model with distinctly named sub-modules for path-based selector tests."""

    def __init__(self) -> None:
        super().__init__()
        self.fc0 = nn.Linear(4, 8)
        self.fc1 = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(torch.relu(self.fc0(x)))


class _ParameterLeaf(nn.Module):
    """Single-parameter leaf module used for selector-based optimizer partitioning."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))


class _SingleParameterModel(nn.Module):
    """Minimal model with one parameter for staged scheduler tests."""

    def __init__(self) -> None:
        super().__init__()
        self.stage_param = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _ConcurrentParameterModel(nn.Module):
    """Minimal model with two named parameter groups for concurrent scheduler tests."""

    def __init__(self) -> None:
        super().__init__()
        self.fc0 = _ParameterLeaf()
        self.fc1 = _ParameterLeaf()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _ManualOptimizerWrapper:
    """Tiny Lightning-style optimizer wrapper used for host-path tests."""

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self._optimizer = optimizer
        self.zero_grad_calls = 0
        self.step_calls = 0
        self.toggle_calls = 0

    @property
    def param_groups(self) -> list[dict[str, object]]:
        return self._optimizer.param_groups

    def zero_grad(self) -> None:
        self.zero_grad_calls += 1
        self._optimizer.zero_grad()

    def step(self, closure=None, **kwargs):  # noqa: ANN001,ANN003
        self.step_calls += 1
        if closure is None:
            return self._optimizer.step(**kwargs)
        return self._optimizer.step(closure=closure, **kwargs)

    @contextmanager
    def toggle_model(self, sync_grad: bool = True):  # noqa: ARG002
        self.toggle_calls += 1
        yield


class _ConstantLossLightningWrapper(StandardLightningWrapper):
    """Standard wrapper variant whose training loss is a constant scalar."""

    def _run_step(self, batch, batch_idx, stage):  # noqa: ANN001,ANN201
        param = next(self.model.parameters(), None)
        device = param.device if param is not None else torch.device("cpu")
        loss = torch.tensor(1.0, device=device, requires_grad=True)
        return loss, 1, batch


class _StageLRCaptureCallback(Callback):
    """Capture staged optimizer learning rates across epoch boundaries."""

    def __init__(self) -> None:
        self.records: list[dict[str, float | int]] = []

    def on_train_epoch_start(self, trainer, pl_module) -> None:  # noqa: ANN001
        controller = pl_module._optimization_controller
        self.records.append(
            {
                "event": 0,
                "epoch": trainer.current_epoch,
                "active_index": controller._program.active_index,
                "stage_0_lr": controller._program.stages[0].optimizer.param_groups[0]["lr"],
                "stage_1_lr": controller._program.stages[1].optimizer.param_groups[0]["lr"],
            }
        )


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_layer_model() -> _TwoLayer:
    """Tiny two-layer model used across optimization tests."""
    return _TwoLayer()


@pytest.fixture
def sequential_two_stage_settings() -> OptimizerPolicySettings:
    """Two sequential stages: AdamW (epoch 5 trigger) → Adam (no trigger)."""
    return OptimizerPolicySettings(
        stages=(
            OptimizationStageSettings(
                optimizer=AdamWSettings(),
                trigger=TriggerSettings(at_epoch=5),
            ),
            OptimizationStageSettings(
                optimizer=AdamSettings(),
            ),
        )
    )


@pytest.fixture
def concurrent_two_optimizer_settings() -> OptimizerPolicySettings:
    """Single concurrent ConcurrentOptimizer: AdamW on fc0, Adam on fc1 (disjoint sets)."""
    return OptimizerPolicySettings(
        default_optimizer=ConcurrentOptimizerSettings(
            optimizers=(AdamWSettings(), AdamSettings()),
            selectors=(
                ParameterSelectorSettings(prefix="fc0"),
                ParameterSelectorSettings(prefix="fc1"),
            ),
        )
    )


@pytest.fixture
def concurrent_with_scheduler_settings() -> OptimizerPolicySettings:
    """Concurrent stage with StepLR scheduler: AdamW on fc0, Adam on fc1 (disjoint sets)."""
    return OptimizerPolicySettings(
        stages=(
            OptimizationStageSettings(
                optimizer=ConcurrentOptimizerSettings(
                    optimizers=(AdamWSettings(lr=0.2), AdamSettings(lr=0.05)),
                    selectors=(
                        ParameterSelectorSettings(prefix="fc0"),
                        ParameterSelectorSettings(prefix="fc1"),
                    ),
                ),
                scheduler=StepLRSettings(step_size=1, gamma=0.5),
            ),
        )
    )


@pytest.fixture
def sequential_two_stage_with_schedulers_settings() -> OptimizerPolicySettings:
    """Two manual stages with per-stage StepLR schedulers.

    Expected behavior: because stage 0 never transitions within the test window,
    only stage 0's scheduler steps. Stage 1 keeps its configured ``lr=0.05``
    unchanged until it becomes the active stage.
    """
    return OptimizerPolicySettings(
        stages=(
            OptimizationStageSettings(
                optimizer=AdamWSettings(lr=0.2),
                scheduler=StepLRSettings(step_size=1, gamma=0.5),
                trigger=TriggerSettings(at_epoch=99),
            ),
            OptimizationStageSettings(
                optimizer=AdamSettings(lr=0.05),
                scheduler=StepLRSettings(step_size=1, gamma=0.5),
            ),
        )
    )


@pytest.fixture
def sequential_two_stage_transition_schedulers_settings() -> OptimizerPolicySettings:
    """Two manual stages that transition after the first epoch.

    Expected behavior: stage 0 decays once and then yields control. Stage 1
    starts from its own configured ``lr=0.05`` when activated; it does not
    inherit stage 0's decayed learning rate.
    """
    return OptimizerPolicySettings(
        stages=(
            OptimizationStageSettings(
                optimizer=AdamWSettings(lr=0.2),
                scheduler=StepLRSettings(step_size=1, gamma=0.5),
                trigger=TriggerSettings(at_epoch=0),
            ),
            OptimizationStageSettings(
                optimizer=AdamSettings(lr=0.05),
                scheduler=StepLRSettings(step_size=1, gamma=0.5),
            ),
        )
    )


@pytest.fixture
def single_stage_default_scheduler_settings() -> OptimizerPolicySettings:
    """Single automatic stage with default scheduler."""
    return OptimizerPolicySettings(
        default_optimizer=AdamWSettings(),
        default_scheduler=ReduceLROnPlateauSettings(patience=2, factor=0.5),
    )


def _make_wrapper(settings: OptimizerPolicySettings) -> StandardLightningWrapper:
    """Build a minimal StandardLightningWrapper with the given optimization settings."""
    components = WrapperComponents(
        loss_fn=nn.MSELoss(),
        val_metric_routes=[],
        test_metric_routes=[],
        optimizer_policy_settings=settings,
        feature_transforms={"x": ModuleList()},
        target_transforms={"y": ModuleList()},
    )
    return StandardLightningWrapper(
        model_settings=ModelComponentSettings(name="_TwoLayer", module_path=_MODULE),
        settings=WrapperComponentSettings(),
        components=components,
        entry_configs=(
            ValueEntry(name="x", data_role=DataRole.FEATURE),
            ValueEntry(name="y", data_role=DataRole.TARGET),
        ),
    )


def _make_scheduler_probe_wrapper(
    settings: OptimizerPolicySettings,
    *,
    model_name: str,
) -> StandardLightningWrapper:
    """Build a constant-loss wrapper that exercises real optimizer wiring only."""
    components = WrapperComponents(
        loss_fn=nn.MSELoss(),
        val_metric_routes=[],
        test_metric_routes=[],
        optimizer_policy_settings=settings,
        feature_transforms={"x": ModuleList()},
        target_transforms={"y": ModuleList()},
    )
    return _ConstantLossLightningWrapper(
        model_settings=ModelComponentSettings(name=model_name, module_path=_MODULE),
        settings=WrapperComponentSettings(),
        components=components,
        entry_configs=(
            ValueEntry(name="x", data_role=DataRole.FEATURE),
            ValueEntry(name="y", data_role=DataRole.TARGET),
        ),
    )


# ---------------------------------------------------------------------------
# _requires_manual_optimization unit tests
# ---------------------------------------------------------------------------


class TestRequiresManualOptimization:
    def test_single_stage_does_not_require_manual(self, two_layer_model: _TwoLayer) -> None:
        settings = OptimizerPolicySettings()
        program = OptimizerPolicyBuilder().build(two_layer_model, settings)
        assert not _requires_manual_optimization(program)

    def test_sequential_two_stages_require_manual(
        self,
        two_layer_model: _TwoLayer,
        sequential_two_stage_settings: OptimizerPolicySettings,
    ) -> None:
        program = OptimizerPolicyBuilder().build(two_layer_model, sequential_two_stage_settings)
        assert _requires_manual_optimization(program)

    def test_concurrent_group_does_not_require_manual(
        self,
        two_layer_model: _TwoLayer,
        concurrent_two_optimizer_settings: OptimizerPolicySettings,
    ) -> None:
        program = OptimizerPolicyBuilder().build(two_layer_model, concurrent_two_optimizer_settings)
        assert not _requires_manual_optimization(program)

    def test_lbfgs_stage_requires_manual(self, two_layer_model: _TwoLayer) -> None:
        # Build the program directly to bypass optimizer-factory kwarg filtering
        # (LBFGS rejects weight_decay; that's a separate factory concern).
        from dlkit.engine.training.optimization.state import ActiveStage, RunningOptimizerPolicy
        from dlkit.engine.training.optimization.triggers import NoTransitionTrigger

        lbfgs_opt = torch.optim.LBFGS(two_layer_model.parameters())
        stage = ActiveStage(
            optimizer=lbfgs_opt,
            scheduler=None,
            trigger=NoTransitionTrigger(),
            stage_index=0,
        )
        program = RunningOptimizerPolicy(stages=(stage,))
        assert _requires_manual_optimization(program)


# ---------------------------------------------------------------------------
# Controller selection tests
# ---------------------------------------------------------------------------


class TestControllerSelection:
    def test_sequential_two_stages_use_manual_controller(
        self,
        sequential_two_stage_settings: OptimizerPolicySettings,
    ) -> None:
        wrapper = _make_wrapper(sequential_two_stage_settings)
        assert isinstance(wrapper._optimization_controller, ManualOptimizationController)
        assert not wrapper.automatic_optimization

    def test_concurrent_group_uses_automatic_controller(
        self,
        concurrent_two_optimizer_settings: OptimizerPolicySettings,
    ) -> None:
        wrapper = _make_wrapper(concurrent_two_optimizer_settings)
        assert isinstance(wrapper._optimization_controller, AutomaticOptimizationController)
        assert wrapper.automatic_optimization

    def test_single_stage_uses_automatic_controller(self) -> None:
        wrapper = _make_wrapper(OptimizerPolicySettings())
        assert isinstance(wrapper._optimization_controller, AutomaticOptimizationController)
        assert wrapper.automatic_optimization


# ---------------------------------------------------------------------------
# configure_optimizers tests
# ---------------------------------------------------------------------------


class TestConfigureOptimizers:
    def test_single_stage_returns_dict(self) -> None:
        wrapper = _make_wrapper(OptimizerPolicySettings())
        config = wrapper.configure_optimizers()
        assert isinstance(config, dict)
        assert "optimizer" in config

    def test_sequential_two_stages_return_list_of_two_optimizers(
        self,
        sequential_two_stage_settings: OptimizerPolicySettings,
    ) -> None:
        wrapper = _make_wrapper(sequential_two_stage_settings)
        config = wrapper.configure_optimizers()
        assert isinstance(config, list)
        assert len(config) == 2  # noqa: PLR2004

    def test_concurrent_single_stage_returns_dict_with_concurrent_optimizer(
        self,
        concurrent_two_optimizer_settings: OptimizerPolicySettings,
    ) -> None:
        wrapper = _make_wrapper(concurrent_two_optimizer_settings)
        config = wrapper.configure_optimizers()
        assert isinstance(config, dict)
        assert "optimizer" in config
        assert isinstance(config["optimizer"], ConcurrentOptimizer)

    def test_concurrent_stage_with_scheduler_preserved_in_config(
        self,
        concurrent_with_scheduler_settings: OptimizerPolicySettings,
    ) -> None:
        """Scheduler attached to a concurrent stage must appear in configure_optimizers output."""
        wrapper = _make_wrapper(concurrent_with_scheduler_settings)
        config = wrapper.configure_optimizers()
        assert isinstance(config, dict)
        assert "optimizer" in config
        assert isinstance(config["optimizer"], ConcurrentOptimizer)
        assert "lr_scheduler" in config

    def test_single_stage_default_scheduler_preserved_in_config(
        self,
        single_stage_default_scheduler_settings: OptimizerPolicySettings,
    ) -> None:
        """Default single-stage scheduler must appear in automatic config output."""
        wrapper = _make_wrapper(single_stage_default_scheduler_settings)
        config = wrapper.configure_optimizers()
        assert isinstance(config, dict)
        assert "optimizer" in config
        assert "lr_scheduler" in config


# ---------------------------------------------------------------------------
# Runtime: both optimizers actually step parameters
# ---------------------------------------------------------------------------


def _make_batch(batch_size: int = 4, dim: int = 4) -> TensorDict:
    return TensorDict(
        {
            "features": TensorDict({"x": torch.randn(batch_size, dim)}, batch_size=[batch_size]),
            "targets": TensorDict({"y": torch.randn(batch_size, dim)}, batch_size=[batch_size]),
        },
        batch_size=[batch_size],
    )


def _make_probe_batch() -> TensorDict:
    """Smallest batch shape needed for fit-level scheduler tests."""
    return TensorDict(
        {
            "features": TensorDict({"x": torch.zeros(1, 1)}, batch_size=[1]),
            "targets": TensorDict({"y": torch.zeros(1, 1)}, batch_size=[1]),
        },
        batch_size=[1],
    )


class _SingleBatchDataset(IterableDataset[TensorDict]):
    """Yield a single pre-batched TensorDict for Lightning fit() regression tests."""

    def __init__(self, batch: TensorDict | None = None) -> None:
        self._batch = batch if batch is not None else _make_batch()

    def __iter__(self):
        yield self._batch


def _identity_collate(batch: TensorDict) -> TensorDict:
    """Preserve a pre-batched TensorDict when DataLoader auto-collation is disabled."""
    return batch


def _passthrough_collate(batch: TensorDict | list[TensorDict]) -> TensorDict:
    match batch:
        case TensorDict():
            return batch
        case [single]:
            return single
        case _:
            raise AssertionError(f"Unexpected collate payload: {type(batch).__name__}")


def _optimizer_lr(settings: object) -> float:
    return cast("float", getattr(settings, "lr"))


def _fit_single_batch_wrapper(
    wrapper: StandardLightningWrapper,
    *,
    max_epochs: int,
    batch: TensorDict | None = None,
    callbacks: list[Callback] | None = None,
) -> None:
    """Run a minimal CPU Trainer.fit loop against one repeated pre-batched sample."""
    trainer = Trainer(
        max_epochs=max_epochs,
        logger=False,
        enable_checkpointing=False,
        limit_train_batches=1,
        limit_val_batches=0,
        enable_progress_bar=False,
        accelerator="cpu",
        devices=1,
        callbacks=callbacks or [],
    )
    dataloader = DataLoader(
        _SingleBatchDataset(batch=batch),
        batch_size=None,
        collate_fn=_passthrough_collate,
    )
    trainer.fit(wrapper, train_dataloaders=dataloader)


class TestBothOptimizersStep:
    def test_concurrent_both_optimizers_update_parameters(
        self,
        concurrent_two_optimizer_settings: OptimizerPolicySettings,
    ) -> None:
        """ConcurrentOptimizer must produce parameter gradient updates for all sub-optimizers."""
        wrapper = _make_wrapper(concurrent_two_optimizer_settings)
        assert isinstance(wrapper._optimization_controller, AutomaticOptimizationController)

        params_before = {name: p.data.clone() for name, p in wrapper.model.named_parameters()}

        controller = wrapper._optimization_controller
        config = controller.configure_optimizers()
        assert isinstance(config, dict)
        opt = config["optimizer"]
        assert isinstance(opt, ConcurrentOptimizer)

        batch = _make_batch()
        opt.zero_grad()
        loss, _, _ = wrapper._run_step(batch, 0, "train")
        loss.backward()
        opt.step()

        params_after = dict(wrapper.model.named_parameters())
        changed = [
            name
            for name, before in params_before.items()
            if not torch.allclose(before, params_after[name].data)
        ]
        assert len(changed) > 0, "No parameters were updated - optimizers did not step"

    def test_concurrent_optimizer_supports_lightning_automatic_closure(
        self,
        concurrent_two_optimizer_settings: OptimizerPolicySettings,
    ) -> None:
        """Automatic optimization must work through Trainer.fit with concurrent optimizers."""
        wrapper = _make_wrapper(concurrent_two_optimizer_settings)
        params_before = {name: p.data.clone() for name, p in wrapper.model.named_parameters()}
        _fit_single_batch_wrapper(wrapper, max_epochs=1)

        params_after = dict(wrapper.model.named_parameters())
        changed = [
            name
            for name, before in params_before.items()
            if not torch.allclose(before, params_after[name].data)
        ]
        assert changed

    def test_sequential_two_stages_step_only_active_optimizer(
        self,
        sequential_two_stage_settings: OptimizerPolicySettings,
    ) -> None:
        """After BUG-1 fix: stage 1 optimizer must NOT update params during stage 0."""
        wrapper = _make_wrapper(sequential_two_stage_settings)
        assert isinstance(wrapper._optimization_controller, ManualOptimizationController)

        controller = wrapper._optimization_controller
        # Both optimizers are registered but only the active stage (index 0) should step.
        assert controller._program.active_index == 0

        params_before = {name: p.data.clone() for name, p in wrapper.model.named_parameters()}

        # Run one manual training step
        batch = _make_batch()
        loss_result = wrapper.training_step(batch, 0)
        assert "loss" in loss_result

        # Parameters must have changed (stage 0 optimizer was active)
        params_after = dict(wrapper.model.named_parameters())
        changed = [
            name
            for name, before in params_before.items()
            if not torch.allclose(before, params_after[name].data)
        ]
        assert len(changed) > 0, "Stage-0 optimizer did not update any parameters"

        # Stage index must still be 0 (trigger fires at epoch 5, not after 1 step)
        assert controller._program.active_index == 0

    def test_manual_wrapper_path_uses_manual_backward_host_api(
        self,
        sequential_two_stage_settings: OptimizerPolicySettings,
    ) -> None:
        """Wrapper manual mode should use the host manual-backward API when available."""
        wrapper = _make_wrapper(sequential_two_stage_settings)
        assert isinstance(wrapper._optimization_controller, ManualOptimizationController)

        wrapped_optimizers = [
            _ManualOptimizerWrapper(stage.optimizer)
            for stage in wrapper._optimization_controller._program.stages
        ]
        manual_backward_calls = {"count": 0}

        def manual_backward(loss: torch.Tensor, *args, **kwargs) -> None:  # noqa: ANN002,ANN003
            manual_backward_calls["count"] += 1
            loss.backward(*args, **kwargs)

        cast("Any", wrapper).optimizers = lambda use_pl_optimizer=True: wrapped_optimizers
        cast("Any", wrapper).manual_backward = manual_backward

        params_before = {name: p.data.clone() for name, p in wrapper.model.named_parameters()}
        result = wrapper.training_step(_make_batch(), 0)

        params_after = dict(wrapper.model.named_parameters())
        changed = [
            name
            for name, before in params_before.items()
            if not torch.allclose(before, params_after[name].data)
        ]

        assert "loss" in result
        assert changed
        assert manual_backward_calls["count"] == 1
        assert wrapped_optimizers[0].step_calls == 1
        assert wrapped_optimizers[0].toggle_calls == 1
        assert wrapped_optimizers[1].step_calls == 0

    def test_stage_advances_when_epoch_trigger_fires(
        self,
        sequential_two_stage_settings: OptimizerPolicySettings,
    ) -> None:
        """Epoch trigger at_epoch=5 must advance program to stage 1 at epoch 5.

        Verifies the full config → program → controller advancement path without
        a Trainer: the controller is called directly (as on_train_epoch_end would).
        """
        wrapper = _make_wrapper(sequential_two_stage_settings)
        controller = wrapper._optimization_controller
        assert isinstance(controller, ManualOptimizationController)
        assert controller._program.active_index == 0

        # Epochs 0-4: trigger should not fire (at_epoch=5)
        for epoch in range(5):
            controller.on_epoch_end(epoch, {})
        assert controller._program.active_index == 0, "Stage advanced too early"

        # Epoch 5: trigger fires → advance to stage 1
        controller.on_epoch_end(5, {})
        assert controller._program.active_index == 1, "Stage did not advance at at_epoch=5"

    def test_sequential_two_stage_fit_steps_only_active_stage_scheduler(
        self,
        sequential_two_stage_with_schedulers_settings: OptimizerPolicySettings,
    ) -> None:
        """Trainer.fit should step only the active stage scheduler in manual mode.

        Expected behavior: with no transition, stage 0 decays from ``0.2`` to
        ``0.05`` over two epochs, while stage 1 remains at its configured
        ``0.05`` because inactive staged schedulers do not step.
        """
        wrapper = _make_scheduler_probe_wrapper(
            sequential_two_stage_with_schedulers_settings,
            model_name="_SingleParameterModel",
        )
        controller = wrapper._optimization_controller
        assert isinstance(controller, ManualOptimizationController)

        stage_0 = controller._program.stages[0]
        stage_1 = controller._program.stages[1]
        assert stage_0.optimizer.param_groups[0]["lr"] == pytest.approx(0.2)
        assert stage_1.optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
        assert stage_1.optimizer.param_groups[0]["lr"] == pytest.approx(
            _optimizer_lr(sequential_two_stage_with_schedulers_settings.stages[1].optimizer)
        )

        _fit_single_batch_wrapper(wrapper, max_epochs=2, batch=_make_probe_batch())

        assert controller._program.active_index == 0
        assert stage_0.optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
        assert stage_1.optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
        assert stage_1.optimizer.param_groups[0]["lr"] == pytest.approx(
            _optimizer_lr(sequential_two_stage_with_schedulers_settings.stages[1].optimizer)
        )

    def test_concurrent_stage_fit_steps_scheduler_for_all_inner_optimizers(
        self,
        concurrent_with_scheduler_settings: OptimizerPolicySettings,
    ) -> None:
        """Trainer.fit must decay LR across all sub-optimizers in a concurrent stage."""
        wrapper = _make_scheduler_probe_wrapper(
            concurrent_with_scheduler_settings,
            model_name="_ConcurrentParameterModel",
        )
        controller = wrapper._optimization_controller
        assert isinstance(controller, AutomaticOptimizationController)

        optimizer = controller._program.current.optimizer
        assert isinstance(optimizer, ConcurrentOptimizer)
        assert optimizer.sub_optimizers[0].param_groups[0]["lr"] == pytest.approx(0.2)
        assert optimizer.sub_optimizers[1].param_groups[0]["lr"] == pytest.approx(0.05)

        _fit_single_batch_wrapper(wrapper, max_epochs=2, batch=_make_probe_batch())

        assert optimizer.sub_optimizers[0].param_groups[0]["lr"] == pytest.approx(0.05)
        assert optimizer.sub_optimizers[1].param_groups[0]["lr"] == pytest.approx(0.0125)

    def test_sequential_transition_activates_stage_1_at_its_configured_lr(
        self,
        sequential_two_stage_transition_schedulers_settings: OptimizerPolicySettings,
    ) -> None:
        """Stage 1 should start from its configured LR and then decay under its scheduler.

        Expected behavior:
        - epoch 0 uses stage 0 at ``lr=0.2``
        - epoch 0 end decays stage 0 to ``0.1`` and advances to stage 1
        - epoch 1 starts with stage 1 still at its configured ``lr=0.05``
        - epoch 1 end decays stage 1 to ``0.025``

        This verifies that a later stage does not inherit the previous stage's
        decayed learning rate on activation.
        """
        wrapper = _make_scheduler_probe_wrapper(
            sequential_two_stage_transition_schedulers_settings,
            model_name="_SingleParameterModel",
        )
        controller = wrapper._optimization_controller
        assert isinstance(controller, ManualOptimizationController)

        callback = _StageLRCaptureCallback()
        _fit_single_batch_wrapper(
            wrapper,
            max_epochs=2,
            batch=_make_probe_batch(),
            callbacks=[callback],
        )

        assert len(callback.records) == 2  # noqa: PLR2004
        epoch_0_start = callback.records[0]
        epoch_1_start = callback.records[1]

        assert epoch_0_start["active_index"] == 0
        assert epoch_0_start["stage_0_lr"] == pytest.approx(0.2)
        assert epoch_0_start["stage_1_lr"] == pytest.approx(0.05)

        assert epoch_1_start["active_index"] == 1
        assert epoch_1_start["stage_0_lr"] == pytest.approx(0.1)
        assert epoch_1_start["stage_0_lr"] == pytest.approx(
            _optimizer_lr(sequential_two_stage_transition_schedulers_settings.stages[0].optimizer)
            * 0.5
        )
        assert epoch_1_start["stage_1_lr"] == pytest.approx(0.05)
        assert epoch_1_start["stage_1_lr"] == pytest.approx(
            _optimizer_lr(sequential_two_stage_transition_schedulers_settings.stages[1].optimizer)
        )

        assert controller._program.active_index == 1
        assert controller._program.stages[0].optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
        assert controller._program.stages[0].optimizer.param_groups[0]["lr"] == pytest.approx(
            _optimizer_lr(sequential_two_stage_transition_schedulers_settings.stages[0].optimizer)
            * 0.5
        )
        assert controller._program.stages[1].optimizer.param_groups[0]["lr"] == pytest.approx(0.025)
        assert controller._program.stages[1].optimizer.param_groups[0]["lr"] == pytest.approx(
            _optimizer_lr(sequential_two_stage_transition_schedulers_settings.stages[1].optimizer)
            * 0.5
        )
