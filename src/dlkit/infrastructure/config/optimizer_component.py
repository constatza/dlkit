"""Typed settings for individual optimizer and scheduler components."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator
from pydantic_settings import SettingsConfigDict

from .core.base_settings import BasicSettings, ComponentSettings
from .optimization_selector import ParameterSelectorSettings


class OptimizerComponentSettings(ComponentSettings):
    """Abstract base class for optimizer settings.

    Provides common identity fields (``name``, ``module_path``) and
    ``get_init_kwargs()`` for the factory layer. Do not instantiate directly;
    use one of the concrete typed subclasses or add a new one for a new optimizer.

    Attributes:
        name: Optimizer class name or callable. Defaults to ``"AdamW"``.
        module_path: Import path for the optimizer. Defaults to ``"torch.optim"``.
    """

    model_config = SettingsConfigDict(extra="allow", arbitrary_types_allowed=True)

    name: str | Callable[..., Any] | dict[str, Any] | None = Field(
        default="AdamW", description="Optimizer name"
    )
    module_path: str | None = Field(
        default="torch.optim", description="Module path to the optimizer"
    )


# ---------------------------------------------------------------------------
# Shared config for all concrete typed optimizer classes:
# extra="forbid"  — unknown kwargs raise ValidationError immediately (no silent drops)
# frozen=True     — immutable after construction (inherited from BasicSettings chain,
#                   but repeated explicitly to guard against Pydantic config merge edge-cases)
# ---------------------------------------------------------------------------
_OPT_CONFIG = SettingsConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class AdamWSettings(OptimizerComponentSettings):
    """Settings for ``torch.optim.AdamW``.

    All fields default to PyTorch's own AdamW defaults so that an
    ``AdamWSettings()`` with no arguments is equivalent to calling
    ``AdamW(params)`` with no keyword arguments.

    Attributes:
        name: Discriminator tag — always ``"AdamW"``.
        module_path: Import path.
        lr: Learning rate.
        betas: Coefficients for computing running averages of gradient and its square.
        eps: Term added to denominator for numerical stability.
        weight_decay: L2 regularization coefficient (AdamW default is 0.01, not 0.0).
        amsgrad: Whether to use the AMSGrad variant.
    """

    model_config = _OPT_CONFIG
    name: Literal["AdamW"] = "AdamW"
    module_path: str | None = "torch.optim"
    lr: float = Field(default=1e-3, gt=0, description="Learning rate")
    betas: tuple[float, float] = Field(default=(0.9, 0.999), description="Beta coefficients")
    eps: float = Field(default=1e-8, gt=0, description="Numerical stability term")
    weight_decay: float = Field(
        default=0.01, ge=0, description="L2 regularization (AdamW default: 0.01)"
    )
    amsgrad: bool = Field(default=False, description="Use AMSGrad variant")


class AdamSettings(OptimizerComponentSettings):
    """Settings for ``torch.optim.Adam``.

    Attributes:
        name: Discriminator tag — always ``"Adam"``.
        module_path: Import path.
        lr: Learning rate.
        betas: Coefficients for computing running averages of gradient and its square.
        eps: Term added to denominator for numerical stability.
        weight_decay: L2 regularization coefficient.
        amsgrad: Whether to use the AMSGrad variant.
    """

    model_config = _OPT_CONFIG
    name: Literal["Adam"] = "Adam"
    module_path: str | None = "torch.optim"
    lr: float = Field(default=1e-3, gt=0, description="Learning rate")
    betas: tuple[float, float] = Field(default=(0.9, 0.999), description="Beta coefficients")
    eps: float = Field(default=1e-8, gt=0, description="Numerical stability term")
    weight_decay: float = Field(default=0.0, ge=0, description="L2 regularization")
    amsgrad: bool = Field(default=False, description="Use AMSGrad variant")


class LBFGSSettings(OptimizerComponentSettings):
    """Settings for ``torch.optim.LBFGS``.

    LBFGS requires a closure and does not accept ``weight_decay``. With
    ``extra="forbid"``, attempting ``LBFGSSettings(weight_decay=0.1)`` raises
    a ``ValidationError`` immediately rather than silently dropping the field.

    Attributes:
        name: Discriminator tag — always ``"LBFGS"``.
        module_path: Import path.
        lr: Learning rate (step size). LBFGS default is ``1.0``.
        max_iter: Maximum number of iterations per optimization step.
        max_eval: Maximum number of function evaluations per step.
        tolerance_grad: Termination tolerance on first-order optimality.
        tolerance_change: Termination tolerance on function value / parameter changes.
        history_size: Update history size.
        line_search_fn: Line search algorithm (``None`` or ``"strong_wolfe"``).
    """

    model_config = _OPT_CONFIG
    name: Literal["LBFGS"] = "LBFGS"
    module_path: str | None = "torch.optim"
    lr: float = Field(default=1.0, gt=0, description="Step size")
    max_iter: int = Field(default=20, gt=0, description="Max iterations per step")
    max_eval: int | None = Field(default=None, description="Max function evaluations per step")
    tolerance_grad: float = Field(
        default=1e-7, gt=0, description="First-order optimality tolerance"
    )
    tolerance_change: float = Field(
        default=1e-9, gt=0, description="Value / parameter change tolerance"
    )
    history_size: int = Field(default=100, gt=0, description="Update history size")
    line_search_fn: str | None = Field(default=None, description="Line search algorithm")


class MuonSettings(OptimizerComponentSettings):
    """Settings for ``torch.optim.Muon``.

    Muon (Momentum + Update Orthogonalization) uses Newton–Schulz iterations
    to orthogonalize the weight update matrix.

    For 1-D and non-matrix parameters, use a separate stage:
    ``OptimizationStageSettings(optimizer=AdamWSettings(...),
    selector=NonMuonSelectorSettings())`` — do not rely on Muon's built-in
    AdamW fallback.

    Attributes:
        name: Discriminator tag — always ``"Muon"``.
        module_path: Import path.
        lr: Learning rate for the Muon update.
        momentum: Momentum coefficient.
        nesterov: Whether to use Nesterov momentum.
        ns_steps: Number of Newton–Schulz iterations.
    """

    model_config = _OPT_CONFIG
    name: Literal["Muon"] = "Muon"
    module_path: str | None = "torch.optim"
    lr: float = Field(default=0.02, gt=0, description="Muon learning rate")
    momentum: float = Field(default=0.95, ge=0, le=1, description="Momentum coefficient")
    nesterov: bool = Field(default=True, description="Use Nesterov momentum")
    ns_steps: int = Field(default=5, gt=0, description="Newton–Schulz iterations")


class ConcurrentOptimizerSettings(BasicSettings):
    """Settings for running multiple optimizers simultaneously on disjoint parameter sets.

    Fits in ``OptimizerSpec`` like any other optimizer — use as ``default_optimizer``
    for the common case (no stages needed) or inside ``OptimizationStageSettings.optimizer``
    when concurrent training is one phase of a sequential program.

    Auto-selector inference: when ``selectors`` is empty and at least one inner
    optimizer is ``MuonSettings``, the builder assigns ``MuonEligibleSelector`` to
    each Muon optimizer and ``NonMuonSelector`` to all others. For any other
    concurrent split (e.g. encoder/decoder by module path) ``selectors`` is required.

    Attributes:
        name: Discriminator tag — always ``"Concurrent"``.
        optimizers: Inner optimizers running concurrently. Required; no default.
        selectors: Per-optimizer parameter selectors (1-to-1 with ``optimizers``).
            Empty tuple requires at least one ``MuonSettings`` in ``optimizers``
            (Muon/non-Muon split is auto-inferred). Provide explicit selectors for
            any other partitioning strategy.
    """

    model_config = _OPT_CONFIG
    name: Literal["Concurrent"] = "Concurrent"
    optimizers: tuple[OptimizerSpec, ...]  # resolved by model_rebuild
    selectors: tuple[ParameterSelectorSettings | None, ...] = Field(
        default=(), description="Per-optimizer selectors; empty = auto-infer (Muon only)"
    )

    @model_validator(mode="after")
    def _validate_selectors(self) -> ConcurrentOptimizerSettings:
        """Enforce selector validity.

        Returns:
            Self after validation.

        Raises:
            ValueError: If selectors are non-empty with wrong length, or empty without Muon.
        """
        if self.selectors and len(self.selectors) != len(self.optimizers):
            raise ValueError(
                f"selectors length ({len(self.selectors)}) must match "
                f"optimizers length ({len(self.optimizers)})."
            )
        has_muon = any(isinstance(opt, MuonSettings) for opt in self.optimizers)
        if not self.selectors and not has_muon:
            raise ValueError(
                "ConcurrentOptimizerSettings with no selectors requires at least one "
                "MuonSettings sub-optimizer (Muon/non-Muon split is auto-inferred). "
                "For other concurrent splits provide explicit selectors=."
            )
        return self


# ---------------------------------------------------------------------------
# Public type alias — use this as the field type wherever an optimizer is configured.
# Pydantic dispatches deserialization to the correct subclass via the ``name`` discriminator.
# ---------------------------------------------------------------------------
OptimizerSpec = Annotated[
    AdamWSettings | AdamSettings | LBFGSSettings | MuonSettings | ConcurrentOptimizerSettings,
    Field(discriminator="name"),
]

# Resolve the recursive OptimizerSpec reference inside ConcurrentOptimizerSettings.
ConcurrentOptimizerSettings.model_rebuild(_types_namespace={"OptimizerSpec": OptimizerSpec})


class SchedulerComponentSettings(ComponentSettings):
    """Settings for a learning rate scheduler component.

    Schedules learning rate adjustments during training. Supports both common
    scheduler names and custom implementations.

    Attributes:
        name: Scheduler name, callable, or dict spec. Defaults to None (no scheduler).
        module_path: Import path for the scheduler. Defaults to "torch.optim.lr_scheduler".
        monitor: Metric to monitor for plateau-based scheduling. Defaults to "val_loss".
        frequency: Update frequency (epoch or step). Defaults to 1.
    """

    model_config = SettingsConfigDict(extra="allow")

    name: str | Callable[..., Any] | dict[str, Any] | None = Field(
        default=None, description="Scheduler name"
    )
    module_path: str | None = Field(
        default="torch.optim.lr_scheduler", description="Module path to the scheduler"
    )
    monitor: str = Field(
        default="val_loss", description="Metric to monitor for learning rate adjustment"
    )
    frequency: int = Field(default=1, description="Update frequency (epochs or steps)")


# ---------------------------------------------------------------------------
# Shared config for all concrete typed scheduler classes:
# extra="forbid"  — unknown kwargs raise ValidationError immediately (no silent drops)
# frozen=True     — immutable after construction
# ---------------------------------------------------------------------------
_SCHED_CONFIG = SettingsConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class StepLRSettings(SchedulerComponentSettings):
    """Settings for ``torch.optim.lr_scheduler.StepLR``.

    Reduces learning rate by a factor (gamma) every step_size epochs.

    Attributes:
        name: Discriminator tag — always ``"StepLR"``.
        module_path: Import path.
        step_size: Period of LR decay (epochs).
        gamma: Multiplicative factor of LR decay.
    """

    model_config = _SCHED_CONFIG
    name: Literal["StepLR"] = "StepLR"
    module_path: str | None = "torch.optim.lr_scheduler"
    step_size: int = Field(default=10, gt=0, description="Period of LR decay (epochs)")
    gamma: float = Field(default=0.1, gt=0, description="Multiplicative decay factor")


class CosineAnnealingLRSettings(SchedulerComponentSettings):
    """Settings for ``torch.optim.lr_scheduler.CosineAnnealingLR``.

    Anneals learning rate using cosine schedule over T_max iterations.

    Attributes:
        name: Discriminator tag — always ``"CosineAnnealingLR"``.
        module_path: Import path.
        T_max: Maximum number of iterations.
        eta_min: Minimum learning rate.
    """

    model_config = _SCHED_CONFIG
    name: Literal["CosineAnnealingLR"] = "CosineAnnealingLR"
    module_path: str | None = "torch.optim.lr_scheduler"
    T_max: int = Field(default=50, gt=0, description="Maximum number of iterations")
    eta_min: float = Field(default=0.0, ge=0, description="Minimum learning rate")


class ReduceLROnPlateauSettings(SchedulerComponentSettings):
    """Settings for ``torch.optim.lr_scheduler.ReduceLROnPlateau``.

    Reduces learning rate when a monitored metric plateaus.

    Note: ``monitor`` and ``frequency`` are inherited from ``SchedulerComponentSettings``
    and excluded from PyTorch constructor kwargs by the factory.

    Attributes:
        name: Discriminator tag — always ``"ReduceLROnPlateau"``.
        mode: Optimization direction (``"min"`` or ``"max"``).
        factor: Factor by which LR is reduced.
        patience: Epochs with no improvement before reduction.
        min_lr: Lower bound on LR.
        threshold: Threshold for measuring improvement.
        cooldown: Epochs to wait after LR reduction before resuming normal operation.
        eps: Minimal decay applied to learning rate.
    """

    model_config = _SCHED_CONFIG
    name: Literal["ReduceLROnPlateau"] = "ReduceLROnPlateau"
    module_path: str | None = "torch.optim.lr_scheduler"
    mode: Literal["min", "max"] = Field(default="min", description="Optimization direction")
    factor: float = Field(default=0.1, gt=0, lt=1, description="Factor by which LR is reduced")
    patience: int = Field(
        default=10, gt=0, description="Epochs with no improvement before reduction"
    )
    min_lr: float = Field(default=0.0, ge=0, description="Lower bound on LR")
    threshold: float = Field(default=1e-4, gt=0, description="Threshold for measuring improvement")
    cooldown: int = Field(default=0, ge=0, description="Cooldown epochs after LR reduction")
    eps: float = Field(default=1e-8, gt=0, description="Minimal decay applied to lr")


class CosineAnnealingWarmRestartsSettings(SchedulerComponentSettings):
    """Settings for ``torch.optim.lr_scheduler.CosineAnnealingWarmRestarts``.

    Anneals learning rate with warm restarts using cosine schedule.

    Attributes:
        name: Discriminator tag — always ``"CosineAnnealingWarmRestarts"``.
        module_path: Import path.
        T_0: Iterations for the first restart.
        T_mult: Multiplier for restart period after each restart.
        eta_min: Minimum learning rate.
    """

    model_config = _SCHED_CONFIG
    name: Literal["CosineAnnealingWarmRestarts"] = "CosineAnnealingWarmRestarts"
    module_path: str | None = "torch.optim.lr_scheduler"
    T_0: int = Field(default=10, gt=0, description="Iterations for first restart")
    T_mult: int = Field(default=1, ge=1, description="Restart period multiplier")
    eta_min: float = Field(default=0.0, ge=0, description="Minimum learning rate")


# ---------------------------------------------------------------------------
# Public type alias — use this as the field type wherever a scheduler is configured.
# Pydantic dispatches deserialization to the correct subclass via the ``name`` discriminator.
# ---------------------------------------------------------------------------
SchedulerSpec = Annotated[
    StepLRSettings
    | CosineAnnealingLRSettings
    | ReduceLROnPlateauSettings
    | CosineAnnealingWarmRestartsSettings,
    Field(discriminator="name"),
]
