"""Training service using the new Orchestrator (Phase 1)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from dlkit.interfaces.api.domain import TrainingResult, WorkflowError
from dlkit.tools.io import provisioning
from dlkit.tools.io import locations
from dlkit.tools.config import GeneralSettings
from dlkit.runtime.workflows.orchestrator import Orchestrator
from dlkit.interfaces.api.overrides.path_context import (
    path_override_context,
    get_current_path_context,
)


class TrainingService:
    """Service for executing training workflows via the Orchestrator."""

    def __init__(self) -> None:
        """Initialize training service."""
        self.service_name = "training_service"

    def execute_training(
        self,
        settings: GeneralSettings,
        checkpoint_path: Path | None = None,
    ) -> TrainingResult:
        """Execute training workflow.

        Args:
            settings: DLKit configuration settings
            checkpoint_path: Optional checkpoint path for resuming

        Returns:
            TrainingResult on success; raises WorkflowError on failure
        """
        start_time = time.time()

        try:
            # Ensure standard run directories exist under the output root
            try:
                provisioning.ensure_run_dirs()
            except Exception:
                pass
            # If settings define a root_dir, apply it as a temporary path context unless already overridden
            overrides: dict[str, Any] = {}
            ctx = get_current_path_context()
            try:
                root_from_cfg = getattr(getattr(settings, "SESSION", None), "root_dir", None)
                if root_from_cfg and not (ctx and getattr(ctx, "root_dir", None)):
                    overrides["root_dir"] = root_from_cfg
            except Exception:
                pass
            # Phase 1 orchestration: build + execute via Orchestrator
            orchestrator = Orchestrator()
            if overrides:
                with path_override_context(overrides):
                    exec_result = orchestrator.execute_training(settings)
            else:
                exec_result = orchestrator.execute_training(settings)
            duration = time.time() - start_time
            value = exec_result
            # Replace duration_seconds with measured duration
            return TrainingResult(
                model_state=getattr(value, "model_state", None),
                metrics=getattr(value, "metrics", None),
                artifacts=getattr(value, "artifacts", None),
                duration_seconds=duration
                if duration > 0
                else getattr(value, "duration_seconds", 0.0),
            )

        except Exception as e:
            if isinstance(e, WorkflowError):
                raise
            raise WorkflowError(
                f"Training execution failed: {str(e)}",
                {"service": self.service_name, "error": str(e)},
            )

    def _extract_metrics(self, model_state) -> dict[str, Any]:
        """Extract training metrics from model state.

        Args:
            model_state: Trained model state

        Returns:
            Dictionary of training metrics
        """
        metrics = {}

        # Extract metrics from trainer if available
        if model_state.trainer:
            trainer = model_state.trainer

            def _collect(source):
                collected: dict[str, Any] = {}
                if not source:
                    return collected
                for key, value in source.items():
                    try:
                        collected[key] = float(value)
                    except Exception:
                        collected[key] = value
                return collected

            metrics.update(_collect(getattr(trainer, "callback_metrics", None)))
            metrics.update(_collect(getattr(trainer, "progress_bar_metrics", None)))
            metrics.update(_collect(getattr(trainer, "logged_metrics", None)))

        # Add model-specific metrics if available
        if hasattr(model_state.model, "metrics"):
            metrics.update(model_state.model.metrics)

        return metrics

    def _collect_artifacts(self, model_state) -> dict[str, Path]:
        """Collect training artifacts.

        Args:
            model_state: Trained model state

        Returns:
            Dictionary of artifact paths
        """
        artifacts = {}

        # Collect checkpoint paths from all ModelCheckpoint callbacks
        if model_state.trainer:
            from lightning.pytorch.callbacks import ModelCheckpoint

            # Check trainer.checkpoint_callback (primary checkpoint callback)
            if hasattr(model_state.trainer, "checkpoint_callback"):
                callback = model_state.trainer.checkpoint_callback
                if callback and isinstance(callback, ModelCheckpoint):
                    if hasattr(callback, "best_model_path") and callback.best_model_path:
                        artifacts["best_checkpoint"] = Path(callback.best_model_path)
                    if hasattr(callback, "last_model_path") and callback.last_model_path:
                        artifacts["last_checkpoint"] = Path(callback.last_model_path)

            # Also check all callbacks for ModelCheckpoint instances
            if hasattr(model_state.trainer, "callbacks"):
                for callback in model_state.trainer.callbacks:
                    if isinstance(callback, ModelCheckpoint):
                        if hasattr(callback, "best_model_path") and callback.best_model_path:
                            artifacts["best_checkpoint"] = Path(callback.best_model_path)
                        if hasattr(callback, "last_model_path") and callback.last_model_path:
                            artifacts["last_checkpoint"] = Path(callback.last_model_path)

                        # Also check filesystem directly if paths aren't set but dirpath is configured
                        if callback.dirpath and not artifacts.get("last_checkpoint"):
                            # Look for checkpoint files in the dirpath
                            dirpath = Path(callback.dirpath)
                            if dirpath.exists():
                                # Check for *-last.ckpt pattern
                                last_checkpoints = list(dirpath.glob("*-last.ckpt"))
                                if last_checkpoints:
                                    artifacts["last_checkpoint"] = last_checkpoints[0]
                                # Also check for *.ckpt without -last suffix as fallback
                                elif not artifacts.get("best_checkpoint"):
                                    ckpt_files = list(dirpath.glob("*.ckpt"))
                                    if ckpt_files:
                                        artifacts["best_checkpoint"] = ckpt_files[0]

        # Collect log directory under standardized output policy
        try:
            log_dir = locations.output("logs")
            if log_dir.exists():
                artifacts["logs"] = log_dir
        except Exception:
            pass

        return artifacts
