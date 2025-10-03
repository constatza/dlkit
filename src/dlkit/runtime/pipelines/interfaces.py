"""Core interfaces for the processing pipeline.

This module defines the abstract interfaces that enable dependency inversion
throughout the processing system. Each interface represents a single
responsibility and can be implemented independently.
"""

from abc import ABC, abstractmethod

import torch

from dlkit.tools.config.data_entries import DataEntry


class DataProvider(ABC):
    """Interface for dataflow provision following Dependency Inversion Principle.

    Data providers are responsible for loading and caching dataflow from various
    sources. The interface allows different implementations for different
    dataflow sources (files, databases, memory, etc.).
    """

    @abstractmethod
    def can_handle(self, entry: DataEntry) -> bool:
        """Check if this provider can handle the given dataflow entry type.

        Args:
            entry: Data entry configuration to check

        Returns:
            True if this provider can handle the entry type
        """
        pass

    @abstractmethod
    def load_data(self, entry: DataEntry, idx: int) -> torch.Tensor:
        """Load dataflow for a specific entry and index.

        Args:
            entry: Data entry configuration
            idx: Index of the dataflow sample to load

        Returns:
            Loaded tensor dataflow for the specified index

        Raises:
            ValueError: If the entry cannot be handled by this provider
        """
        pass

    @abstractmethod
    def get_length(self, entry: DataEntry) -> int:
        """Get the total length of dataflow for this entry.

        Args:
            entry: Data entry configuration

        Returns:
            Total number of samples available
        """
        pass


class BatchProcessor(ABC):
    """Interface for batch processing using Strategy Pattern.

    Batch processors handle specific types of batch transformations
    and can be swapped out without affecting the rest of the pipeline.
    """

    @abstractmethod
    def process(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Process a batch of

        Args:
            batch: Dictionary mapping dataflow names to tensors

        Returns:
            Processed batch dictionary
        """
        pass


class ModelInvoker(ABC):
    """Interface for model invocation using Command Pattern.

    Model invokers encapsulate the command of calling a model with
    specific inputs. This allows different models to be called in
    different ways without the calling code needing to know the specifics.
    """

    @abstractmethod
    def invoke(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Invoke the model with the provided features.

        Args:
            features: Dictionary of feature tensors to feed to the model

        Returns:
            Dictionary of model outputs

        Raises:
            RuntimeError: If model invocation fails
        """
        pass

    @abstractmethod
    def get_expected_inputs(self) -> list[str]:
        """Get the list of expected input names for this model.

        Returns:
            List of input names the model expects
        """
        pass

    @abstractmethod
    def get_output_names(self) -> list[str]:
        """Get the list of output names this model produces.

        Returns:
            List of output names the model produces
        """
        pass


class OutputClassifier(ABC):
    """Interface for classifying model outputs using Strategy Pattern.

    Output classifiers determine which model outputs are latents
    (intermediate representations) and which are predictions (matching targets).
    Different strategies can be used based on naming, shapes, or configuration.
    """

    @abstractmethod
    def classify(
        self, model_outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Classify model outputs into latents and predictions.

        Args:
            model_outputs: Raw outputs from model invocation
            targets: Target dataflow for comparison

        Returns:
            Tuple of (latents, predictions) dictionaries
        """
        pass


class OutputNamer(ABC):
    """Interface for post-classification prediction key naming.

    This interface decouples the concerns of "what is a prediction" (classification)
    from "what should the prediction key be" (naming). Implementations may use
    different strategies (shape matching, explicit config, heuristics) to map
    prediction keys to target names without affecting latent/prediction decisions.
    """

    @abstractmethod
    def rename_predictions(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        *,
        model_outputs: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return a new predictions dict with keys possibly renamed.

        Args:
            predictions: Predictions classified from model outputs (keys are model output names)
            targets: Available target tensors keyed by target name
            model_outputs: Optional full model outputs, if needed by the strategy

        Returns:
            A new dictionary of predictions with keys renamed according to the strategy.
        """
        pass
