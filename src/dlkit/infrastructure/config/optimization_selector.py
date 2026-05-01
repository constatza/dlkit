"""Typed settings for parameter selector configurations.

The public ``ParameterSelectorSettings`` class covers the common cases (role or module
path prefix) without requiring a discriminator field.

Advanced composed selectors are available as separate classes for power users.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator
from pydantic_settings import SettingsConfigDict

from .core.base_settings import BasicSettings

_SELECTOR_CONFIG = SettingsConfigDict(extra="forbid", frozen=True)


class ParameterSelectorSettings(BasicSettings):
    """Common parameter selector — set role OR prefix, not both.

    Attributes:
        role: Parameter role name to match (e.g. ``"muon_eligible"``).
        prefix: Module path prefix (e.g. ``"encoder"`` matches ``encoder.weight``).
    """

    model_config = _SELECTOR_CONFIG
    role: str | None = Field(default=None, description="Parameter role name to match")
    prefix: str | None = Field(default=None, description="Module path prefix (startswith match)")

    @model_validator(mode="after")
    def _validate_exactly_one(self) -> ParameterSelectorSettings:
        """Enforce that exactly one of role or prefix is specified.

        Returns:
            Self after validation.

        Raises:
            ValueError: If both or neither of role and prefix are set.
        """
        both_set = self.role is not None and self.prefix is not None
        neither_set = self.role is None and self.prefix is None
        if both_set or neither_set:
            raise ValueError("Specify exactly one of role or prefix.")
        return self


# ---------------------------------------------------------------------------
# Advanced composite selectors — set-algebra combinators (internal / power users)
# ---------------------------------------------------------------------------

_AdvancedSelectorSpec = Annotated[
    "RoleSelectorSettings | ModulePathSelectorSettings | MuonEligibleSelectorSettings | NonMuonSelectorSettings | IntersectionSelectorSettings | UnionSelectorSettings | DifferenceSelectorSettings",
    Field(discriminator="kind"),
]


class RoleSelectorSettings(BasicSettings):
    """Select parameters by semantic role (advanced/internal).

    Attributes:
        kind: Discriminator tag — always ``"role"``.
        role: Parameter role name to match.
    """

    model_config = _SELECTOR_CONFIG
    kind: Literal["role"] = "role"
    role: str = Field(..., description="Parameter role name to match")


class ModulePathSelectorSettings(BasicSettings):
    """Select parameters by module path prefix (advanced/internal).

    Attributes:
        kind: Discriminator tag — always ``"module_path"``.
        prefix: Module path prefix.
    """

    model_config = _SELECTOR_CONFIG
    kind: Literal["module_path"] = "module_path"
    prefix: str = Field(..., description="Module path prefix (startswith match)")


class MuonEligibleSelectorSettings(BasicSettings):
    """Select parameters eligible for the Muon optimizer (advanced/internal).

    Attributes:
        kind: Discriminator tag — always ``"muon_eligible"``.
    """

    model_config = _SELECTOR_CONFIG
    kind: Literal["muon_eligible"] = "muon_eligible"


class NonMuonSelectorSettings(BasicSettings):
    """Select parameters NOT eligible for Muon (advanced/internal).

    Attributes:
        kind: Discriminator tag — always ``"non_muon"``.
    """

    model_config = _SELECTOR_CONFIG
    kind: Literal["non_muon"] = "non_muon"


class _CompositeSelectorBase(BasicSettings):
    """Internal base for set-algebra selectors."""

    model_config = _SELECTOR_CONFIG
    children: tuple[_AdvancedSelectorSpec, ...] = Field(
        default=(), description="Child selectors combined by the set operation"
    )


class IntersectionSelectorSettings(_CompositeSelectorBase):
    """Select parameters satisfying ALL child selectors.

    Attributes:
        kind: Discriminator tag — always ``"intersection"``.
    """

    kind: Literal["intersection"] = "intersection"


class UnionSelectorSettings(_CompositeSelectorBase):
    """Select parameters satisfying ANY child selector.

    Attributes:
        kind: Discriminator tag — always ``"union"``.
    """

    kind: Literal["union"] = "union"


class DifferenceSelectorSettings(BasicSettings):
    """Select parameters satisfying include but NOT exclude.

    Attributes:
        kind: Discriminator tag — always ``"difference"``.
        include: Selector for parameters to include.
        exclude: Selector for parameters to exclude.
    """

    model_config = _SELECTOR_CONFIG
    kind: Literal["difference"] = "difference"
    include: _AdvancedSelectorSpec
    exclude: _AdvancedSelectorSpec


# Resolve forward references in composite selectors.
_CompositeSelectorBase.model_rebuild()
DifferenceSelectorSettings.model_rebuild()
