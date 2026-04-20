"""Typed settings for parameter selector configurations.

Each selector variant is a focused, closed class — no nullable fields that belong
to other variants. The ``ParameterSelectorSettings`` alias is the discriminated union
used as the public field type.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from .core.base_settings import BasicSettings

_SELECTOR_CONFIG = SettingsConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Leaf selectors — no recursive children
# ---------------------------------------------------------------------------


class RoleSelectorSettings(BasicSettings):
    """Select parameters by semantic role.

    Attributes:
        kind: Discriminator tag — always ``"role"``.
        role: Parameter role name to match (e.g. ``"encoder"``, ``"head"``).
    """

    model_config = _SELECTOR_CONFIG
    kind: Literal["role"] = "role"
    role: str = Field(..., description="Parameter role name to match")


class ModulePathSelectorSettings(BasicSettings):
    """Select parameters by module path prefix.

    Attributes:
        kind: Discriminator tag — always ``"module_path"``.
        prefix: Module path prefix (e.g. ``"encoder"`` matches ``encoder.weight``, ``encoder.bias``).
    """

    model_config = _SELECTOR_CONFIG
    kind: Literal["module_path"] = "module_path"
    prefix: str = Field(..., description="Module path prefix (startswith match)")


class MuonEligibleSelectorSettings(BasicSettings):
    """Select parameters eligible for the Muon optimizer.

    Muon eligibility requires hidden-layer 2-D weight matrices (``ndim == 2``).

    Attributes:
        kind: Discriminator tag — always ``"muon_eligible"``.
    """

    model_config = _SELECTOR_CONFIG
    kind: Literal["muon_eligible"] = "muon_eligible"


class NonMuonSelectorSettings(BasicSettings):
    """Select parameters that are NOT eligible for the Muon optimizer.

    Covers embeddings, biases, 1-D tensors, and head / first-layer parameters.

    Attributes:
        kind: Discriminator tag — always ``"non_muon"``.
    """

    model_config = _SELECTOR_CONFIG
    kind: Literal["non_muon"] = "non_muon"


# ---------------------------------------------------------------------------
# Composite selectors — set-algebra combinators with recursive children
# ---------------------------------------------------------------------------


class _CompositeSelectorBase(BasicSettings):
    """Internal base for set-algebra selectors.

    Shares the ``children`` field. Not part of the public API; use the
    concrete subclasses directly.

    Attributes:
        children: Child selectors combined by the set operation.
    """

    model_config = _SELECTOR_CONFIG
    children: tuple[ParameterSelectorSettings, ...] = Field(
        default=(), description="Child selectors combined by the set operation"
    )


class IntersectionSelectorSettings(_CompositeSelectorBase):
    """Select parameters that satisfy ALL child selectors.

    Attributes:
        kind: Discriminator tag — always ``"intersection"``.
        children: Selectors whose results are intersected.
    """

    kind: Literal["intersection"] = "intersection"


class UnionSelectorSettings(_CompositeSelectorBase):
    """Select parameters that satisfy ANY child selector.

    Attributes:
        kind: Discriminator tag — always ``"union"``.
        children: Selectors whose results are unioned.
    """

    kind: Literal["union"] = "union"


class DifferenceSelectorSettings(BasicSettings):
    """Select parameters that satisfy ``include`` but NOT ``exclude``.

    Attributes:
        kind: Discriminator tag — always ``"difference"``.
        include: Selector for parameters to include.
        exclude: Selector for parameters to exclude from the include set.
    """

    model_config = _SELECTOR_CONFIG
    kind: Literal["difference"] = "difference"
    include: ParameterSelectorSettings
    exclude: ParameterSelectorSettings


# ---------------------------------------------------------------------------
# Public discriminated union alias
# ---------------------------------------------------------------------------

ParameterSelectorSettings = Annotated[
    RoleSelectorSettings
    | ModulePathSelectorSettings
    | MuonEligibleSelectorSettings
    | NonMuonSelectorSettings
    | IntersectionSelectorSettings
    | UnionSelectorSettings
    | DifferenceSelectorSettings,
    Field(discriminator="kind"),
]
"""Discriminated union of all selector variants.

Pydantic dispatches deserialization to the correct subclass via the ``kind``
discriminator field. Use ``ParameterSelectorSettings | None`` as a field type
where a selector is optional.
"""

# Resolve forward references to ParameterSelectorSettings.
# The annotations are deferred (from __future__ import annotations), so
# model_rebuild() resolves them now that the alias is defined.
_CompositeSelectorBase.model_rebuild()
DifferenceSelectorSettings.model_rebuild()
