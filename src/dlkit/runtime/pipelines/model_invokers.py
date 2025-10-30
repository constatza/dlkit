"""Model invokers implementing the Command Pattern.

This module provides different strategies for invoking models with feature
Each invoker encapsulates the command of calling a specific type of model,
allowing the calling code to remain agnostic about model-specific details.
"""

from typing import Any

import torch
from torch import nn

from .interfaces import ModelInvoker


class StandardModelInvoker(ModelInvoker):
    """Standard model invocation for simple forward pass models.

    This invoker handles models that accept either:
    1. A single tensor input (uses first feature)
    2. Multiple positional arguments (unpacks features by order)
    3. Keyword arguments (passes features as **kwargs)

    Attributes:
        _model: The PyTorch model to invoke
        _input_mode: How to pass inputs to the model ('single', 'args', 'kwargs')
        _expected_inputs: List of expected input names in order
    """

    def __init__(
        self, model: nn.Module, input_mode: str = "kwargs", expected_inputs: list[str] = None
    ):
        """Initialize the standard model invoker.

        Args:
            model: PyTorch model to invoke
            input_mode: How to pass inputs ('single', 'args', 'kwargs')
            expected_inputs: List of expected input names (for 'args' mode)
        """
        self._model = model
        self._input_mode = input_mode
        self._expected_inputs = expected_inputs or []

    @property
    def model(self) -> nn.Module:
        """Get the underlying PyTorch model.

        Returns:
            The PyTorch nn.Module being invoked
        """
        return self._model

    def invoke(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Invoke the model with the provided features.

        Args:
            features: Dictionary of feature tensors

        Returns:
            Dictionary of model outputs (normalized)

        Raises:
            RuntimeError: If model invocation fails
        """
        try:
            # Extract inputs based on mode
            if self._input_mode == "single":
                # Use first feature as single input
                input_tensor = next(iter(features.values()))
                outputs = self._model(input_tensor)
            elif self._input_mode == "args":
                # Pass features as positional arguments in specified order
                args = [features[name] for name in self._expected_inputs if name in features]
                outputs = self._model(*args)
            else:  # 'kwargs'
                # Pass features as keyword arguments
                outputs = self._model(**features)

            return self._normalize_outputs(outputs)

        except Exception as e:
            raise RuntimeError(f"Model invocation failed: {e}") from e

    def get_expected_inputs(self) -> list[str]:
        """Get the list of expected input names for this model.

        Returns:
            List of input names the model expects
        """
        return self._expected_inputs.copy()

    def get_output_names(self) -> list[str]:
        """Get the list of output names this model produces.

        Returns:
            List of output names (determined dynamically)
        """
        # For standard models, output names are determined at runtime
        return ["output"]  # Default name, overridden by _normalize_outputs

    def _normalize_outputs(self, outputs: Any) -> dict[str, torch.Tensor]:
        """Normalize model outputs to a consistent dictionary format.

        Args:
            outputs: Raw model outputs

        Returns:
            Dictionary mapping output names to tensors
        """
        if isinstance(outputs, torch.Tensor):
            # Single tensor output
            return {"output": outputs}
        elif isinstance(outputs, (tuple, list)):
            # Multiple tensor outputs
            return {f"output_{i}": tensor for i, tensor in enumerate(outputs)}
        elif isinstance(outputs, dict):
            # Dictionary outputs (already normalized)
            return outputs
        else:
            raise ValueError(f"Unsupported output type: {type(outputs)}")


class MultiInputModelInvoker(ModelInvoker):
    """Model invoker for models with multiple named inputs.

    This invoker is designed for models that explicitly expect
    multiple named inputs and provides clear mapping between
    feature names and model input parameters.

    Attributes:
        _model: The PyTorch model to invoke
        _input_mapping: Mapping from feature names to model parameter names
    """

    def __init__(self, model: nn.Module, input_mapping: dict[str, str] = None):
        """Initialize the multi-input model invoker.

        Args:
            model: PyTorch model to invoke
            input_mapping: Mapping from feature names to model parameter names
                          If None, uses direct mapping (feature name = parameter name)
        """
        self._model = model
        self._input_mapping = input_mapping or {}

    @property
    def model(self) -> nn.Module:
        """Get the underlying PyTorch model.

        Returns:
            The PyTorch nn.Module being invoked
        """
        return self._model

    def invoke(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Invoke the model with mapped feature inputs.

        Args:
            features: Dictionary of feature tensors

        Returns:
            Dictionary of model outputs

        Raises:
            RuntimeError: If model invocation fails
        """
        try:
            # Map feature names to model parameter names
            model_inputs = {}
            for feature_name, tensor in features.items():
                param_name = self._input_mapping.get(feature_name, feature_name)
                model_inputs[param_name] = tensor

            outputs = self._model(**model_inputs)
            return self._normalize_outputs(outputs)

        except Exception as e:
            raise RuntimeError(f"Multi-input model invocation failed: {e}") from e

    def get_expected_inputs(self) -> list[str]:
        """Get the list of expected input names for this model.

        Returns:
            List of feature names expected by this invoker
        """
        # Return feature names (not model parameter names)
        return list(self._input_mapping.keys()) if self._input_mapping else []

    def get_output_names(self) -> list[str]:
        """Get the list of output names this model produces.

        Returns:
            List of output names (determined dynamically)
        """
        return ["output"]  # Default, overridden by runtime inspection

    def _normalize_outputs(self, outputs: Any) -> dict[str, torch.Tensor]:
        """Normalize model outputs to dictionary format.

        Args:
            outputs: Raw model outputs

        Returns:
            Dictionary mapping output names to tensors
        """
        if isinstance(outputs, torch.Tensor):
            return {"output": outputs}
        elif isinstance(outputs, dict):
            return outputs
        elif isinstance(outputs, (tuple, list)):
            return {f"output_{i}": tensor for i, tensor in enumerate(outputs)}
        else:
            raise ValueError(f"Unsupported output type: {type(outputs)}")


class SequenceModelInvoker(ModelInvoker):
    """Model invoker for sequence models (RNNs, Transformers).

    This invoker handles models that work with sequential dataflow
    and may require specific input formatting or attention masks.

    Attributes:
        _model: The sequence model to invoke
        _sequence_key: Name of the feature containing sequence dataflow
        _attention_key: Name of the feature containing attention masks (optional)
    """

    def __init__(
        self,
        model: nn.Module,
        sequence_key: str = "input_ids",
        attention_key: str = "attention_mask",
    ):
        """Initialize the sequence model invoker.

        Args:
            model: Sequence model to invoke
            sequence_key: Feature name containing sequence dataflow
            attention_key: Feature name containing attention masks
        """
        self._model = model
        self._sequence_key = sequence_key
        self._attention_key = attention_key

    @property
    def model(self) -> nn.Module:
        """Get the underlying PyTorch model.

        Returns:
            The PyTorch nn.Module being invoked
        """
        return self._model

    def invoke(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Invoke the sequence model with appropriate inputs.

        Args:
            features: Dictionary of feature tensors

        Returns:
            Dictionary of model outputs

        Raises:
            RuntimeError: If model invocation fails or required features are missing
        """
        if self._sequence_key not in features:
            raise RuntimeError(f"Required sequence feature '{self._sequence_key}' not found")

        try:
            # Prepare model inputs
            model_inputs = {self._sequence_key: features[self._sequence_key]}

            # Add attention mask if available
            if self._attention_key in features:
                model_inputs[self._attention_key] = features[self._attention_key]

            outputs = self._model(**model_inputs)
            return self._normalize_outputs(outputs)

        except Exception as e:
            raise RuntimeError(f"Sequence model invocation failed: {e}") from e

    def get_expected_inputs(self) -> list[str]:
        """Get the list of expected input names for this model.

        Returns:
            List of input names the sequence model expects
        """
        return [self._sequence_key, self._attention_key]

    def get_output_names(self) -> list[str]:
        """Get the list of output names this model produces.

        Returns:
            List of output names (common sequence model outputs)
        """
        return ["logits", "hidden_states", "attentions"]

    def _normalize_outputs(self, outputs: Any) -> dict[str, torch.Tensor]:
        """Normalize sequence model outputs.

        Args:
            outputs: Raw model outputs (often has .logits, .hidden_states, etc.)

        Returns:
            Dictionary mapping output names to tensors
        """
        if hasattr(outputs, "logits"):
            # Transformer-style outputs
            result = {"logits": outputs.logits}
            if hasattr(outputs, "hidden_states"):
                result["hidden_states"] = outputs.hidden_states
            if hasattr(outputs, "attentions"):
                result["attentions"] = outputs.attentions
            return result
        elif isinstance(outputs, torch.Tensor):
            return {"output": outputs}
        elif isinstance(outputs, dict):
            return outputs
        else:
            raise ValueError(f"Unsupported sequence model output type: {type(outputs)}")


class ModelInvokerFactory:
    """Factory for creating appropriate model invokers.

    This factory uses simple heuristics to determine the best invoker type
    for a given model, following the Factory Pattern.
    """

    @staticmethod
    def create_invoker(model: nn.Module, model_type: str = "auto", **kwargs) -> ModelInvoker:
        """Create an appropriate model invoker for the given model.

        Args:
            model: PyTorch model to create an invoker for
            model_type: Type of invoker ("auto", "standard", "multi_input", "sequence")
            **kwargs: Additional arguments for specific invoker types

        Returns:
            Appropriate ModelInvoker instance
        """
        if model_type == "auto":
            model_type = ModelInvokerFactory._detect_model_type(model)

        if model_type == "sequence":
            return SequenceModelInvoker(model, **kwargs)
        elif model_type == "multi_input":
            return MultiInputModelInvoker(model, **kwargs)
        else:  # "standard"
            # Heuristic: if forward takes a single non-self argument, use 'single' mode
            try:
                import inspect

                sig = inspect.signature(model.forward)
                params = [
                    p
                    for p in sig.parameters.values()
                    if p.name != "self"
                    and p.kind
                    in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                ]
                if len(params) == 1:
                    return StandardModelInvoker(model, input_mode="single", **kwargs)
            except Exception:
                pass
            return StandardModelInvoker(model, **kwargs)

    @staticmethod
    def _detect_model_type(model: nn.Module) -> str:
        """Detect the appropriate model type based on model characteristics.

        Args:
            model: PyTorch model to analyze

        Returns:
            Detected model type string
        """
        model_name = model.__class__.__name__.lower()

        # Check for sequence models
        if any(
            seq_indicator in model_name
            for seq_indicator in ["bert", "gpt", "transformer", "lstm", "gru"]
        ):
            return "sequence"

        # Default to standard
        return "standard"
