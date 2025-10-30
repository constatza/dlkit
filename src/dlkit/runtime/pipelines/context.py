"""Processing context for dataflow flow through the pipeline.

This module defines the context object that carries dataflow through the
processing pipeline steps. It implements a clean separation of concerns
by organizing dataflow into logical categories.
"""

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ProcessingContext:
    """Context object carrying dataflow through the processing pipeline.

    This context follows the Context pattern, providing a shared state
    that can be passed through the Chain of Responsibility processing steps.
    Each step can read from and write to the appropriate sections.

    The context organizes dataflow into logical categories:
    - raw_batch: Original dataflow from the dataset
    - features: Input dataflow for the model
    - targets: Ground truth dataflow
    - model_outputs: Raw outputs from model invocation
    - latents: Intermediate representations from the model
    - predictions: Model outputs that match target shapes/names
    - loss_data: Data required for loss computation

    Attributes:
        raw_batch: Original batch dataflow from dataset
        features: Feature dataflow to feed to model
        targets: Target/ground truth dataflow
        model_outputs: Raw model outputs before classification
        latents: Intermediate model representations
        predictions: Model predictions corresponding to targets
        loss_data: Combined dataflow for loss computation
        artifacts: Auxiliary objects attached by pipeline steps (e.g., original graph batches)
    """

    # Input dataflow
    raw_batch: dict[str, torch.Tensor] = field(default_factory=dict)

    # Separated dataflow components
    features: dict[str, torch.Tensor] = field(default_factory=dict)
    targets: dict[str, torch.Tensor] = field(default_factory=dict)

    # Model processing results
    model_outputs: dict[str, torch.Tensor] = field(default_factory=dict)
    latents: dict[str, torch.Tensor] = field(default_factory=dict)
    predictions: dict[str, torch.Tensor] = field(default_factory=dict)

    # Final processed dataflow
    loss_data: dict[str, torch.Tensor] = field(default_factory=dict)

    # Auxiliary attachments for pipeline-specific data (e.g., original graph batches)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def clear(self) -> None:
        """Clear all dataflow from the context.

        Useful for resetting the context between batches or for memory management.
        """
        self.raw_batch.clear()
        self.features.clear()
        self.targets.clear()
        self.model_outputs.clear()
        self.latents.clear()
        self.predictions.clear()
        self.loss_data.clear()
        self.artifacts.clear()

    def get_all_data(self) -> dict[str, torch.Tensor]:
        """Get all available dataflow combined into a single dictionary.

        This method combines features, targets, latents, and predictions
        into a single dictionary for cases where all dataflow is needed.

        Returns:
            Combined dictionary of all available dataflow
        """
        result = {}
        result.update(self.features)
        result.update(self.targets)
        result.update(self.latents)
        result.update(self.predictions)
        return result

    def has_model_outputs(self) -> bool:
        """Check if model outputs are available.

        Returns:
            True if model_outputs contains dataflow
        """
        return bool(self.model_outputs)

    def has_classifications(self) -> bool:
        """Check if outputs have been classified into latents/predictions.

        Returns:
            True if either latents or predictions contain dataflow
        """
        return bool(self.latents) or bool(self.predictions)
