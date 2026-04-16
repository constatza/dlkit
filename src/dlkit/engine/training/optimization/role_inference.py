"""Parameter role inference strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch.nn as nn

from dlkit.domain.nn.parameter_roles import ParameterRole
from dlkit.domain.nn.role_provider import IParameterRoleProvider


class IParameterRoleInferenceStrategy(ABC):
    """Abstract strategy for inferring the role of a parameter.

    A strategy examines a parameter (by name, shape, and module context) and
    either returns a classified ParameterRole or defers (returns None) to
    allow other strategies in a composite chain to make a classification.
    """

    @abstractmethod
    def infer(self, model: nn.Module, name: str, parameter: nn.Parameter) -> ParameterRole | None:
        """Infer the role of a parameter.

        Args:
            model: The neural network module containing the parameter.
            name: Fully-qualified parameter name from named_parameters().
            parameter: The actual parameter tensor.

        Returns:
            A ParameterRole if this strategy can classify, None to defer.
        """
        ...


class _RoleProviderAdapter(IParameterRoleInferenceStrategy):
    """Adapter that wraps an IParameterRoleProvider to use as a strategy.

    Allows models that implement IParameterRoleProvider to participate in
    the composite inference strategy chain.
    """

    def __init__(self, provider: IParameterRoleProvider) -> None:
        """Initialize the adapter.

        Args:
            provider: A model implementing IParameterRoleProvider.
        """
        self._provider = provider

    def infer(self, model: nn.Module, name: str, parameter: nn.Parameter) -> ParameterRole | None:
        """Look up the parameter role in the provider's role dictionary.

        Args:
            model: The neural network module (unused; provider has its own model).
            name: Fully-qualified parameter name.
            parameter: The parameter tensor (unused; lookup is by name).

        Returns:
            The role if found in provider.parameter_roles(), None otherwise.
        """
        roles = self._provider.parameter_roles()
        return roles.get(name)


class GenericModuleRoleInferenceStrategy(IParameterRoleInferenceStrategy):
    """Strategy for classifying parameters by generic naming conventions.

    Applies heuristics that work for standard PyTorch modules:
    - Parameters named "bias" or ending with ".bias" → BIAS
    - Parameters with "norm" in the name (case-insensitive) → NORMALIZATION
    - All other parameters → None (deferred to other strategies)
    """

    def infer(self, model: nn.Module, name: str, parameter: nn.Parameter) -> ParameterRole | None:
        """Apply generic naming heuristics.

        Args:
            model: The neural network module (unused).
            name: Fully-qualified parameter name.
            parameter: The parameter tensor (unused).

        Returns:
            BIAS for bias-like names, NORMALIZATION for norm-like names,
            None otherwise.
        """
        # Check for bias
        if name == "bias" or name.endswith(".bias"):
            return ParameterRole.BIAS

        # Check for normalization
        if "norm" in name.lower():
            return ParameterRole.NORMALIZATION

        return None


class FFNNRoleInferenceStrategy(IParameterRoleInferenceStrategy):
    """Strategy for classifying feed-forward neural network parameters.

    Inspects the model's parameter structure to identify positional roles:
    - First weight matrix → INPUT
    - Last weight matrix → OUTPUT
    - Middle weight matrices → HIDDEN
    - Bias and normalization parameters are left for other strategies.

    Construction pre-computes the sorted weight names so inference is O(1).
    """

    def __init__(self, model: nn.Module) -> None:
        """Initialize the strategy by analyzing the model structure.

        Args:
            model: The neural network module to analyze.
        """
        self._model = model
        # Extract weight-only parameter names (exclude bias and norm)
        weight_names: list[str] = []
        for name, _param in model.named_parameters():
            # Skip bias and normalization parameters
            if "bias" in name.lower() or "norm" in name.lower():
                continue
            weight_names.append(name)

        # Sort for consistent ordering
        weight_names.sort()
        self._weight_names = weight_names

    def infer(self, model: nn.Module, name: str, parameter: nn.Parameter) -> ParameterRole | None:
        """Classify a weight parameter by its position in the layer sequence.

        Args:
            model: The neural network module (unused; uses precomputed state).
            name: Fully-qualified parameter name.
            parameter: The parameter tensor (unused).

        Returns:
            INPUT if the weight is first, OUTPUT if last, HIDDEN if middle,
            None for bias/normalization or non-weight parameters.
        """
        # Let other strategies handle bias and normalization
        if "bias" in name.lower() or "norm" in name.lower():
            return None

        # Only classify if this is a known weight
        if name not in self._weight_names:
            return None

        # Get position in the weight list
        idx = self._weight_names.index(name)
        num_weights = len(self._weight_names)

        if num_weights == 0:
            return None
        if num_weights == 1:
            return ParameterRole.OUTPUT
        if idx == 0:
            return ParameterRole.INPUT
        if idx == num_weights - 1:
            return ParameterRole.OUTPUT
        return ParameterRole.HIDDEN


class AutoencoderRoleInferenceStrategy(IParameterRoleInferenceStrategy):
    """Strategy for classifying autoencoder and seq2seq parameters.

    Identifies encoder and decoder submodules based on naming conventions.
    """

    def infer(self, model: nn.Module, name: str, parameter: nn.Parameter) -> ParameterRole | None:
        """Classify by encoder/decoder naming.

        Args:
            model: The neural network module (unused).
            name: Fully-qualified parameter name.
            parameter: The parameter tensor (unused).

        Returns:
            ENCODER if name contains "encoder", DECODER if contains "decoder",
            None otherwise.
        """
        if "encoder" in name.lower():
            return ParameterRole.ENCODER
        if "decoder" in name.lower():
            return ParameterRole.DECODER
        return None


class GraphRoleInferenceStrategy(IParameterRoleInferenceStrategy):
    """Strategy for classifying graph neural network parameters.

    Identifies embedding-like parameters by naming convention.
    """

    def infer(self, model: nn.Module, name: str, parameter: nn.Parameter) -> ParameterRole | None:
        """Classify by embedding naming.

        Args:
            model: The neural network module (unused).
            name: Fully-qualified parameter name.
            parameter: The parameter tensor (unused).

        Returns:
            EMBEDDING if name contains "embed", None otherwise.
        """
        if "embed" in name.lower():
            return ParameterRole.EMBEDDING
        return None


class CompositeParameterRoleInferenceStrategy(IParameterRoleInferenceStrategy):
    """Composite strategy that chains multiple inference strategies.

    Tries each strategy in order until one returns a non-None role.
    Falls back to UNKNOWN if all strategies defer.
    """

    def __init__(self, strategies: Sequence[IParameterRoleInferenceStrategy]) -> None:
        """Initialize the composite strategy.

        Args:
            strategies: Sequence of strategies to try in order.
        """
        self._strategies = list(strategies)

    def infer(self, model: nn.Module, name: str, parameter: nn.Parameter) -> ParameterRole | None:
        """Try each strategy in sequence.

        Args:
            model: The neural network module.
            name: Fully-qualified parameter name.
            parameter: The parameter tensor.

        Returns:
            First non-None result from any strategy, or UNKNOWN if all defer.
        """
        for strategy in self._strategies:
            role = strategy.infer(model, name, parameter)
            if role is not None:
                return role
        return ParameterRole.UNKNOWN


def make_default_inference_strategy(
    model: nn.Module,
) -> CompositeParameterRoleInferenceStrategy:
    """Build the default composite role inference strategy.

    The default chain is:
    1. IParameterRoleProvider check (if model implements it)
    2. FFNNRoleInferenceStrategy (position-based for feed-forward nets)
    3. GraphRoleInferenceStrategy (embedding detection)
    4. AutoencoderRoleInferenceStrategy (encoder/decoder detection)
    5. GenericModuleRoleInferenceStrategy (bias, normalization, unknown)

    Args:
        model: The neural network module to build a strategy for.

    Returns:
        A composite strategy ready to infer roles for the model's parameters.
    """
    strategies: list[IParameterRoleInferenceStrategy] = []

    # Check if model provides its own roles
    if isinstance(model, IParameterRoleProvider):
        strategies.append(_RoleProviderAdapter(model))

    # Add domain-specific strategies
    strategies.extend(
        [
            FFNNRoleInferenceStrategy(model),
            GraphRoleInferenceStrategy(),
            AutoencoderRoleInferenceStrategy(),
            GenericModuleRoleInferenceStrategy(),
        ]
    )

    return CompositeParameterRoleInferenceStrategy(strategies)
