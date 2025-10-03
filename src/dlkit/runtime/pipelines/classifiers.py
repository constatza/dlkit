"""Output classification strategies using the Strategy Pattern.

This module provides different strategies for classifying model outputs into
latents (intermediate representations) and predictions (outputs that match targets).
Each strategy implements a different approach to determine the classification.
"""

import torch

from dlkit.tools.config.data_entries import Latent, Prediction
from .interfaces import OutputClassifier


class ShapeBasedClassifier(OutputClassifier):
    """Classify model outputs based on tensor shapes matching targets.

    This classifier uses shape matching to determine if a model output
    corresponds to a target (prediction) or is an intermediate representation (latent).
    Outputs with shapes matching targets are classified as predictions.

    Attributes:
        _shape_tolerance: Tolerance for shape matching (default: exact match)
    """

    def __init__(self, shape_tolerance: float = 0.0):
        """Initialize the shape-based classifier.

        Args:
            shape_tolerance: Tolerance for shape differences (0.0 = exact match)
        """
        self._shape_tolerance = shape_tolerance

    def classify(
        self, model_outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Classify outputs based on shape matching with targets.

        Args:
            model_outputs: Dictionary of raw model outputs
            targets: Dictionary of target tensors for shape comparison

        Returns:
            Tuple of (latents, predictions) dictionaries
        """
        if not targets:
            # No targets available - all outputs are latents
            return model_outputs.copy(), {}

        predictions = {}
        latents = {}
        target_shapes = {name: tensor.shape for name, tensor in targets.items()}

        for output_name, output_tensor in model_outputs.items():
            output_shape = output_tensor.shape

            # Decide only whether this is a prediction; do not rename keys here
            matches_target = self._matches_any_target_shape(output_shape, target_shapes)

            if matches_target:
                predictions[output_name] = output_tensor
            else:
                latents[output_name] = output_tensor

        return latents, predictions

    def _matches_any_target_shape(
        self, output_shape: torch.Size, target_shapes: dict[str, torch.Size]
    ) -> bool:
        """Check if output shape matches any target shape.

        Args:
            output_shape: Shape of the model output
            target_shapes: Dictionary of target shapes to compare against

        Returns:
            True if output shape matches any target shape
        """
        for target_shape in target_shapes.values():
            if self._shapes_match(output_shape, target_shape):
                return True
        return False

    def _shapes_match(self, shape1: torch.Size, shape2: torch.Size) -> bool:
        """Check if two shapes match within tolerance.

        Args:
            shape1: First shape to compare
            shape2: Second shape to compare

        Returns:
            True if shapes match within tolerance
        """
        if len(shape1) != len(shape2):
            return False

        if self._shape_tolerance == 0.0:
            return shape1 == shape2

        # With tolerance, check each dimension
        for dim1, dim2 in zip(shape1, shape2):
            if abs(dim1 - dim2) > self._shape_tolerance * max(dim1, dim2):
                return False

        return True


class NameBasedClassifier(OutputClassifier):
    """Classify model outputs based on naming conventions.

    This classifier uses naming patterns to determine output classification.
    Outputs with names matching certain patterns are classified as predictions,
    while others are classified as latents.

    Attributes:
        _prediction_patterns: List of patterns indicating predictions
        _latent_patterns: List of patterns indicating latents
        _target_names: Set of target names for direct matching
    """

    def __init__(self, prediction_patterns: list[str] = None, latent_patterns: list[str] = None):
        """Initialize the name-based classifier.

        Args:
            prediction_patterns: List of name patterns for predictions
            latent_patterns: List of name patterns for latents
        """
        self._prediction_patterns = prediction_patterns or [
            "_pred",
            "_prediction",
            "output",
            "logits",
            "scores",
        ]
        self._latent_patterns = latent_patterns or [
            "_latent",
            "_hidden",
            "_embedding",
            "_features",
            "attention",
        ]

    def classify(
        self, model_outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Classify outputs based on naming patterns.

        Args:
            model_outputs: Dictionary of raw model outputs
            targets: Dictionary of target tensors (for name matching)

        Returns:
            Tuple of (latents, predictions) dictionaries
        """
        predictions = {}
        latents = {}
        target_names = set(targets.keys()) if targets else set()

        for output_name, output_tensor in model_outputs.items():
            if self._is_prediction(output_name, target_names):
                predictions[output_name] = output_tensor
            else:
                latents[output_name] = output_tensor

        return latents, predictions

    def _is_prediction(self, output_name: str, target_names: set[str]) -> bool:
        """Determine if an output name indicates a prediction.

        Args:
            output_name: Name of the model output
            target_names: Set of target names for direct matching

        Returns:
            True if output name indicates a prediction
        """
        output_lower = output_name.lower()

        # Check direct match with target names
        if output_name in target_names:
            return True

        # Check prediction patterns
        for pattern in self._prediction_patterns:
            if pattern.lower() in output_lower:
                return True

        # Check latent patterns (if matches latent pattern, it's not a prediction)
        for pattern in self._latent_patterns:
            if pattern.lower() in output_lower:
                return False

        # Default: if no clear pattern, assume prediction
        return True


class ConfigBasedClassifier(OutputClassifier):
    """Classify model outputs based on explicit configuration.

    This classifier uses explicit configuration objects to determine
    the classification of model outputs. This provides the most control
    and clarity about output classification.

    Attributes:
        _latent_configs: Dictionary mapping latent names to configurations
        _prediction_configs: Dictionary mapping prediction names to configurations
    """

    def __init__(
        self, latent_configs: list[Latent] = None, prediction_configs: list[Prediction] = None
    ):
        """Initialize the config-based classifier.

        Args:
            latent_configs: List of latent dataflow entry configurations
            prediction_configs: List of prediction dataflow entry configurations
        """
        self._latent_configs = {config.name: config for config in (latent_configs or [])}
        self._prediction_configs = {config.name: config for config in (prediction_configs or [])}

    def classify(
        self, model_outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Classify outputs based on explicit configuration.

        Args:
            model_outputs: Dictionary of raw model outputs
            targets: Dictionary of target tensors (used for prediction mapping)

        Returns:
            Tuple of (latents, predictions) dictionaries
        """
        predictions = {}
        latents = {}

        for output_name, output_tensor in model_outputs.items():
            if output_name in self._latent_configs:
                latents[output_name] = output_tensor
            elif output_name in self._prediction_configs:
                predictions[output_name] = output_tensor
            else:
                # No explicit configuration - use fallback logic
                if self._is_likely_prediction(output_name, targets):
                    predictions[output_name] = output_tensor
                else:
                    latents[output_name] = output_tensor

        return latents, predictions

    def _is_likely_prediction(self, output_name: str, targets: dict[str, torch.Tensor]) -> bool:
        """Fallback logic for outputs not in explicit configuration.

        Args:
            output_name: Name of the model output
            targets: Dictionary of target tensors

        Returns:
            True if output is likely a prediction
        """
        # Check if name matches any target name
        if output_name in targets:
            return True

        # Check common prediction patterns
        prediction_indicators = ["output", "pred", "logits", "scores"]
        output_lower = output_name.lower()

        return any(indicator in output_lower for indicator in prediction_indicators)

    def add_latent_config(self, config: Latent) -> None:
        """Add a latent configuration.

        Args:
            config: Latent dataflow entry configuration to add
        """
        self._latent_configs[config.name] = config

    def add_prediction_config(self, config: Prediction) -> None:
        """Add a prediction configuration.

        Args:
            config: Prediction dataflow entry configuration to add
        """
        self._prediction_configs[config.name] = config


class HybridClassifier(OutputClassifier):
    """Hybrid classifier combining multiple classification strategies.

    This classifier applies multiple strategies in sequence, allowing
    different approaches to be combined for more robust classification.

    Attributes:
        _classifiers: List of classifiers to apply in order
        _voting_strategy: How to resolve conflicts between classifiers
    """

    def __init__(self, classifiers: list[OutputClassifier], voting_strategy: str = "first_match"):
        """Initialize the hybrid classifier.

        Args:
            classifiers: List of classifiers to apply
            voting_strategy: Strategy for resolving conflicts ("first_match", "majority")
        """
        self._classifiers = classifiers
        self._voting_strategy = voting_strategy

    def classify(
        self, model_outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Classify outputs using multiple strategies.

        Args:
            model_outputs: Dictionary of raw model outputs
            targets: Dictionary of target tensors

        Returns:
            Tuple of (latents, predictions) dictionaries
        """
        if not self._classifiers:
            return model_outputs.copy(), {}

        if self._voting_strategy == "first_match":
            return self._first_match_strategy(model_outputs, targets)
        elif self._voting_strategy == "majority":
            return self._majority_vote_strategy(model_outputs, targets)
        else:
            raise ValueError(f"Unknown voting strategy: {self._voting_strategy}")

    def _first_match_strategy(
        self, model_outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Use the first classifier that provides a classification.

        Args:
            model_outputs: Dictionary of raw model outputs
            targets: Dictionary of target tensors

        Returns:
            Tuple of (latents, predictions) from first successful classifier
        """
        for classifier in self._classifiers:
            try:
                latents, predictions = classifier.classify(model_outputs, targets)
                if latents or predictions:  # At least one classification made
                    return latents, predictions
            except Exception:
                continue  # Try next classifier

        # Fallback: all outputs as latents
        return model_outputs.copy(), {}

    def _majority_vote_strategy(
        self, model_outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Use majority voting across classifiers.

        Args:
            model_outputs: Dictionary of raw model outputs
            targets: Dictionary of target tensors

        Returns:
            Tuple of (latents, predictions) based on majority vote
        """
        votes = {}  # output_name -> {"latent": count, "prediction": count}

        for classifier in self._classifiers:
            try:
                latents, predictions = classifier.classify(model_outputs, targets)

                for output_name in model_outputs:
                    if output_name not in votes:
                        votes[output_name] = {"latent": 0, "prediction": 0}

                    if output_name in latents:
                        votes[output_name]["latent"] += 1
                    elif output_name in predictions:
                        votes[output_name]["prediction"] += 1

            except Exception:
                continue  # Skip failed classifier

        # Classify based on majority vote
        predictions = {}
        latents = {}

        for output_name, output_tensor in model_outputs.items():
            vote_counts = votes.get(output_name, {"latent": 0, "prediction": 0})

            if vote_counts["prediction"] > vote_counts["latent"]:
                predictions[output_name] = output_tensor
            else:
                latents[output_name] = output_tensor

        return latents, predictions
