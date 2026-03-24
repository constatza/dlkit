"""Optuna implementation of hyperparameter optimization."""

from __future__ import annotations

from typing import Any
from collections.abc import Callable
import optuna

from dlkit.interfaces.api.domain import WorkflowError
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.core.context import BuildContext
from dlkit.tools.config.core.factories import FactoryProvider
from dlkit.tools.config.samplers.optuna_sampler import create_settings_sampler

from .interfaces import IHyperparameterOptimizer, IOptimizationResult


class OptunaOptimizationResult(IOptimizationResult):
    """Optuna implementation of optimization result."""

    def __init__(self, optuna_trial: Any, study: Any):
        self._trial = optuna_trial
        self._study = study

    @property
    def best_params(self) -> dict[str, Any]:
        """Best hyperparameters from Optuna trial."""
        return dict(self._trial.params) if hasattr(self._trial, "params") else {}

    @property
    def best_value(self) -> float:
        """Best objective value from Optuna trial."""
        return float(self._trial.value)

    @property
    def trial_number(self) -> int:
        """Best trial number from Optuna."""
        return int(self._trial.number)

    @property
    def study_summary(self) -> dict[str, Any]:
        """Optuna study summary."""
        return {
            "trials": len(self._study.trials),
            "direction": self._study.direction.name,
            "study_name": self._study.study_name,
        }


class OptunaOptimizer(IHyperparameterOptimizer):
    """Optuna implementation of hyperparameter optimizer following DIP."""

    def __init__(self):
        self._optuna = optuna

    def optimize(
        self,
        objective: Callable[[Any], float],
        settings: GeneralSettings,
        n_trials: int,
        direction: str = "minimize",
    ) -> IOptimizationResult:
        """Run Optuna hyperparameter optimization.

        Args:
            objective: Objective function taking Optuna trial
            settings: Base settings for optimization configuration
            n_trials: Number of trials to run
            direction: "minimize" or "maximize"

        Returns:
            Optimization result with best trial information
        """
        optuna = self._optuna
        opt_cfg = getattr(settings, "OPTUNA", None)

        if not opt_cfg or not getattr(opt_cfg, "enabled", False):
            raise WorkflowError(
                "Optuna optimization requested but not enabled in configuration",
                {"stage": "optuna"},
            )

        # Create sampler and pruner, passing session seed for injection
        session_seed = settings.SESSION.seed if settings and hasattr(settings, "SESSION") else None
        sampler, pruner = self._build_sampler_pruner(opt_cfg, session_seed)

        # Derive study configuration
        study_name = self._derive_study_name(opt_cfg, settings)
        storage = getattr(opt_cfg, "storage", None)

        # Create and run study
        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
            storage=storage,
            load_if_exists=bool(study_name and storage),
        )

        study.optimize(objective, n_trials=n_trials)

        return OptunaOptimizationResult(study.best_trial, study)

    def create_sampled_settings(
        self, base_settings: GeneralSettings, trial: Any
    ) -> GeneralSettings:
        """Create settings with Optuna-sampled hyperparameters using SettingsSampler.

        Args:
            base_settings: Base configuration settings
            trial: Optuna trial object

        Returns:
            Settings with sampled hyperparameters applied using dedicated sampler
        """
        # Use dedicated SettingsSampler following SRP
        opt_cfg = getattr(base_settings, "OPTUNA", None)
        settings_sampler = create_settings_sampler(opt_cfg)
        return settings_sampler.sample(trial, base_settings)

    def apply_best_params(
        self, base_settings: GeneralSettings, best_params: dict[str, Any]
    ) -> GeneralSettings:
        """Apply best parameters to base settings.

        Args:
            base_settings: Base configuration settings
            best_params: Best hyperparameters to apply

        Returns:
            Settings with best parameters applied
        """
        try:
            if base_settings.MODEL is not None and best_params:
                best_model = base_settings.MODEL.patch(best_params)
                return base_settings.patch({"MODEL": best_model})
        except Exception:
            pass
        return base_settings

    def _build_sampler_pruner(
        self, opt_cfg: Any, session_seed: int | None = None
    ) -> tuple[Any | None, Any | None]:
        """Build Optuna sampler and pruner from configuration.

        Args:
            opt_cfg: Optuna configuration with sampler/pruner settings
            session_seed: Session seed to use if sampler seed is not specified

        Returns:
            Tuple of (sampler, pruner) instances
        """
        sampler = None
        pruner = None
        try:
            ctx = BuildContext(mode="training")

            # Build sampler with session seed injection if needed
            if hasattr(opt_cfg, "sampler") and opt_cfg.sampler:
                # Check if sampler has a seed configured
                sampler_seed = getattr(opt_cfg.sampler, "seed", None)

                # Inject session seed if sampler seed is None
                if sampler_seed is None and session_seed is not None:
                    ctx.overrides["seed"] = session_seed

                sampler = FactoryProvider.create_component(opt_cfg.sampler, ctx)

            # Build pruner (no seed injection needed)
            if hasattr(opt_cfg, "pruner") and opt_cfg.pruner:
                pruner = FactoryProvider.create_component(
                    opt_cfg.pruner, BuildContext(mode="training")
                )
        except Exception:
            pass
        return sampler, pruner

    def _derive_study_name(self, opt_cfg: Any, settings: GeneralSettings) -> str | None:
        """Derive study name from configuration."""
        name = getattr(opt_cfg, "study_name", None)
        if name:
            return str(name)
        try:
            sess = getattr(settings, "SESSION", None)
            if sess and getattr(sess, "name", None):
                return str(sess.name)
        except Exception:
            pass
        return None
