"""Parameter inventory for neural network modules."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

from dlkit.domain.nn.parameter_roles import ParameterRole


@dataclass(frozen=True, slots=True, kw_only=True)
class ParameterDescriptor:
    """Frozen descriptor for a single trainable parameter.

    Attributes:
        name: Fully-qualified parameter name from model.named_parameters().
        parameter: The actual torch.nn.Parameter tensor.
        module_path: Dotted path to the owning module (name without last segment).
        shape: Shape of the parameter tensor.
        ndim: Number of dimensions (rank) of the parameter.
        role: Semantic role classification of this parameter.
    """

    name: str
    parameter: nn.Parameter
    module_path: str
    shape: torch.Size
    ndim: int
    role: ParameterRole = ParameterRole.UNKNOWN


def _module_path_from_name(name: str) -> str:
    """Extract module path from a fully-qualified parameter name.

    Args:
        name: Fully-qualified parameter name from named_parameters().
            Example: "encoder.layer.0.weight" → "encoder.layer.0"

    Returns:
        Module path (everything before the last dot), or empty string if no dot.
    """
    if "." not in name:
        return ""
    return name.rsplit(".", 1)[0]


class IParameterInventory:
    """Abstract interface for enumerating model parameters.

    A parameter inventory is responsible for inspecting a neural network module
    and producing a complete, ordered list of parameter descriptors.
    """

    def list_parameters(self) -> tuple[ParameterDescriptor, ...]:
        """List all trainable parameters in the model.

        Returns:
            Tuple of parameter descriptors in the order returned by
            model.named_parameters(). This order is consistent across
            runs and should be preserved for reproducibility.
        """
        raise NotImplementedError


class TorchParameterInventory(IParameterInventory):
    """Concrete inventory for standard PyTorch nn.Module instances.

    Builds parameter descriptors by inspecting model.named_parameters()
    and optionally classifying each parameter using a role resolver function.

    Attributes:
        _model: The neural network module being inventoried.
        _role_resolver: Optional callable to classify parameter roles.
        _parameters: Cached list of parameter descriptors.
    """

    def __init__(
        self,
        model: nn.Module,
        role_resolver: Callable[[ParameterDescriptor], ParameterRole] | None = None,
    ) -> None:
        """Initialize the inventory.

        Args:
            model: The neural network module to inventory.
            role_resolver: Optional callable that accepts a ParameterDescriptor
                (with role=UNKNOWN) and returns a resolved ParameterRole.
                If None, all parameters retain role=UNKNOWN.
        """
        self._model = model
        self._role_resolver = role_resolver
        self._parameters: tuple[ParameterDescriptor, ...] | None = None

    def list_parameters(self) -> tuple[ParameterDescriptor, ...]:
        """List all trainable parameters in the model.

        Builds descriptors by iterating model.named_parameters(), computing
        shapes, and optionally resolving roles via the role_resolver.

        Returns:
            Tuple of parameter descriptors in named_parameters() order.
        """
        if self._parameters is not None:
            return self._parameters

        descriptors: list[ParameterDescriptor] = []

        for name, param in self._model.named_parameters():
            descriptor = ParameterDescriptor(
                name=name,
                parameter=param,
                module_path=_module_path_from_name(name),
                shape=param.shape,
                ndim=param.ndim,
                role=ParameterRole.UNKNOWN,
            )

            # Resolve role if a resolver is available
            if self._role_resolver is not None:
                resolved_role = self._role_resolver(descriptor)
                descriptor = ParameterDescriptor(
                    name=descriptor.name,
                    parameter=descriptor.parameter,
                    module_path=descriptor.module_path,
                    shape=descriptor.shape,
                    ndim=descriptor.ndim,
                    role=resolved_role,
                )

            descriptors.append(descriptor)

        self._parameters = tuple(descriptors)
        return self._parameters
