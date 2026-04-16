"""Typed settings for parameter selector configurations."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .core.base_settings import BasicSettings

type SelectorKind = Literal[
    "role", "module_path", "muon_eligible", "non_muon", "intersection", "union", "difference"
]


class ParameterSelectorSettings(BasicSettings):
    """Settings for selecting parameter subsets within optimization stages.

    Supports multiple selection strategies: by parameter role, module path pattern,
    muon eligibility, and set operations on selections.

    Attributes:
        kind: Selection strategy (e.g., "role", "module_path", "intersection").
        role: Parameter role name (used when kind="role").
        pattern: Glob/prefix pattern for module paths (used when kind="module_path").
        children: Child selectors for set operations (union, intersection, difference).
    """

    kind: SelectorKind = Field(
        ..., description="Selection strategy: role, module_path, muon_eligible, etc."
    )
    role: str | None = Field(
        default=None, description="Parameter role name (used with kind='role')"
    )
    pattern: str | None = Field(
        default=None,
        description="Glob/prefix pattern for module paths (used with kind='module_path')",
    )
    children: tuple[ParameterSelectorSettings, ...] = Field(
        default=(),
        description="Child selectors for set operations (union, intersection, difference)",
    )
