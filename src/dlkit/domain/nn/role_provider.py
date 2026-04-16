"""Protocol for models that self-annotate their parameter roles."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from dlkit.domain.nn.parameter_roles import ParameterRole


@runtime_checkable
class IParameterRoleProvider(Protocol):
    """Optional protocol a model implements to declare its own parameter roles.

    When a model implements this protocol, the optimization engine uses the
    declared roles directly instead of running inference strategies.

    Example:
        class MyModel(nn.Module, IParameterRoleProvider):
            def parameter_roles(self) -> dict[str, ParameterRole]:
                return {
                    "encoder.weight": ParameterRole.ENCODER,
                    "decoder.weight": ParameterRole.DECODER,
                    "head.weight": ParameterRole.OUTPUT,
                }
    """

    def parameter_roles(self) -> dict[str, ParameterRole]:
        """Return a mapping from parameter name to semantic role.

        Returns:
            Dict mapping fully-qualified parameter names (as returned by
            ``named_parameters()``) to their ``ParameterRole``.
        """
        ...
