"""Inference service using BuildFactory for clear train/infer distinction (Phase 1)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from dlkit.interfaces.api.domain import InferenceResult, WorkflowError
from dlkit.tools.config import GeneralSettings
from dlkit.runtime.workflows.factories.build_factory import BuildFactory
from dlkit.interfaces.api.overrides.path_context import (
    path_override_context,
    get_current_path_context,
)


class InferenceService:
    """Service for executing inference workflows via the BuildFactory."""

    def __init__(self) -> None:
        """Initialize inference service."""
        self.service_name = "inference_service"

    def execute_inference(
        self,
        settings: GeneralSettings,
        checkpoint_path: Path,
    ) -> InferenceResult:
        """Execute inference workflow.

        Args:
            settings: DLKit configuration settings
            checkpoint_path: Path to model checkpoint

        Returns:
            InferenceResult on success; raises WorkflowError on failure
        """
        start_time = time.time()

        try:
            # Apply settings-defined root_dir if present and not already overridden
            overrides: dict[str, Any] = {}
            ctx = get_current_path_context()
            try:
                root_from_cfg = getattr(getattr(settings, "SESSION", None), "root_dir", None)
                if root_from_cfg and not (ctx and getattr(ctx, "root_dir", None)):
                    overrides["root_dir"] = root_from_cfg
            except Exception:
                pass

            # Build components (trainer will be None in inference mode)
            build_factory = BuildFactory()
            if overrides:
                with path_override_context(overrides):
                    components = build_factory.build_components(settings)
            else:
                components = build_factory.build_components(settings)

            # If a checkpoint is provided, validate and attempt to load it
            # into the underlying model in a best-effort manner.
            if checkpoint_path is not None:
                try:
                    from pathlib import Path as _P

                    cp = _P(checkpoint_path)
                    if not cp.exists():
                        raise WorkflowError(
                            f"Checkpoint not found: {checkpoint_path}",
                            {"service": self.service_name, "checkpoint": str(checkpoint_path)},
                        )
                    import torch

                    # For dlkit checkpoints, use weights_only=False since they may contain
                    # custom dlkit settings classes that are trusted
                    state = torch.load(cp, map_location="cpu", weights_only=False)
                    # Support both plain state_dict or lightning-style {'state_dict': ...}
                    if (
                        isinstance(state, dict)
                        and "state_dict" in state
                        and isinstance(state["state_dict"], dict)
                    ):
                        state_dict = state["state_dict"]
                    else:
                        state_dict = state
                    # Best-effort load; ignore missing/unexpected keys
                    try:
                        components.model.load_state_dict(state_dict, strict=False)
                    except Exception:
                        # As a fallback, attempt to remove common prefixes like 'model.'
                        if isinstance(state_dict, dict):
                            stripped = {k.split("model.", 1)[-1]: v for k, v in state_dict.items()}
                            components.model.load_state_dict(stripped, strict=False)
                except WorkflowError:
                    # Bubble up explicit workflow errors
                    raise
                except Exception as e:  # Corrupted or unreadable checkpoint
                    raise WorkflowError(
                        f"Failed to load checkpoint: {checkpoint_path}",
                        {"service": self.service_name, "error": str(e)},
                    )

            # Execute actual inference
            predictions = self._run_inference(components)

            # Create inference result
            duration = time.time() - start_time
            inference_result = InferenceResult(
                model_state=None,
                predictions=predictions,
                metrics=None,
                duration_seconds=duration,
            )

            return inference_result

        except Exception as e:
            if isinstance(e, WorkflowError):
                raise
            raise WorkflowError(
                f"Inference execution failed: {str(e)}",
                {"service": self.service_name, "error": str(e)},
            )

    def _run_inference(self, components) -> Any:
        """Run actual inference using the model.

        Args:
            model_state: Loaded model state

        Returns:
            Model predictions
        """
        model = components.model
        datamodule = components.datamodule

        # Use trainer for prediction if available
        if components.trainer:
            predictions = components.trainer.predict(model, datamodule=datamodule)
            if predictions is None:
                predictions = []
        else:
            # Inference-only: create a lightweight Trainer and use predict_dataloader()
            from lightning.pytorch import Trainer

            if datamodule is not None and hasattr(datamodule, "predict_dataloader"):
                trainer = Trainer(logger=False, enable_checkpointing=False)
                predictions = trainer.predict(model, datamodule=datamodule)
                if predictions is None:
                    predictions = []
            else:
                # No datamodule: cannot build batches automatically. Caller should supply
                # a predict_dataloader via CLI/layer above. Here, return empty.
                predictions = []

        return predictions
