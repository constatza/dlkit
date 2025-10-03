"""Simple prediction strategy using Lightning framework.

This module provides the prediction strategy that uses the existing
Lightning-based inference system for validation and testing scenarios
where the full training infrastructure is available.
"""

from __future__ import annotations

from pathlib import Path

from dlkit.tools.config import GeneralSettings
from dlkit.interfaces.api.domain.models import InferenceResult
from dlkit.interfaces.api.services.inference_service import InferenceService as LegacyInferenceService


class SimplePredictionStrategy:
    """Simple prediction strategy using Lightning framework.

    This strategy wraps the existing Lightning-based inference system
    for scenarios where full training configuration is available and
    validation/testing workflows are needed.
    """

    def __init__(self) -> None:
        """Initialize simple prediction strategy."""
        self._legacy_service = LegacyInferenceService()

    def predict(
        self,
        training_settings: GeneralSettings,
        checkpoint_path: Path | str,
        **overrides
    ) -> InferenceResult:
        """Execute prediction-mode inference using Lightning.

        Args:
            training_settings: General configuration settings
            checkpoint_path: Path to model checkpoint
            **overrides: Additional parameter overrides

        Returns:
            InferenceResult: Inference execution result

        Note:
            This delegates to the existing Lightning-based inference service
            which requires full training configuration and datasets.
        """
        return self._legacy_service.execute_inference(
            training_settings,
            Path(checkpoint_path)
        )