"""Training service using the new Orchestrator (Phase 1)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from dlkit.interfaces.api.domain import TrainingResult, WorkflowError
from dlkit.tools.config import GeneralSettings
from dlkit.tools.utils.logging_config import get_logger
from dlkit.runtime.workflows.orchestrator import Orchestrator
from dlkit.interfaces.api.overrides.path_context import (
    path_override_context,
    get_current_path_context,
)

logger = get_logger(__name__)


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
            # Extract root_dir from settings BEFORE any directory creation
            overrides: dict[str, Any] = {}
            ctx = get_current_path_context()
            try:
                root_from_cfg = getattr(getattr(settings, "SESSION", None), "root_dir", None)
                if root_from_cfg and not (ctx and getattr(ctx, "root_dir", None)):
                    overrides["root_dir"] = root_from_cfg
            except Exception as e:
                logger.debug(f"Failed to extract root_dir from settings (non-fatal): {e}")

            # Establish path context for the training run
            # Directories are created lazily when files are actually written
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
