"""Orchestrators for complex inference workflows.

High-level coordinators that compose multiple use cases
for end-to-end inference scenarios.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from dlkit.interfaces.api.domain.models import InferenceResult

from .use_cases import InferenceUseCase
from ..inputs.inference_input import InferenceInput


class InferenceOrchestrator:
    """High-level orchestrator for inference workflows.

    Provides a simplified interface that coordinates multiple use cases
    for complex inference scenarios.
    """

    def __init__(self, inference_use_case: InferenceUseCase):
        """Initialize with inference use case."""
        self._inference_use_case = inference_use_case

    def infer_from_checkpoint(
        self,
        checkpoint_path: Path | str,
        inputs: InferenceInput | dict[str, Any] | Any,
        device: str = "auto",
        batch_size: int = 32,
        apply_transforms: bool = True
    ) -> InferenceResult:
        """Execute inference from checkpoint with automatic model reconstruction.

        This is the main entry point for the improved inference API.

        Args:
            checkpoint_path: Path to trained model checkpoint
            inputs: Input data (will be wrapped in InferenceInput if needed)
            device: Device specification ("auto", "cpu", "cuda", "mps")
            batch_size: Batch size for processing
            apply_transforms: Whether to apply fitted transforms

        Returns:
            InferenceResult: Complete inference result

        Raises:
            WorkflowError: If inference fails at any step
        """
        # Convert inputs to InferenceInput if needed
        if not isinstance(inputs, InferenceInput):
            inputs = InferenceInput(inputs)

        # Execute inference using the use case
        return self._inference_use_case.execute_inference(
            checkpoint_path=Path(checkpoint_path),
            inputs=inputs,
            device=device,
            batch_size=batch_size,
            apply_transforms=apply_transforms
        )