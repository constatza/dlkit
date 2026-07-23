"""Generic multi-run sweep orchestration — reusable across convergence, ensemble, ablation."""

from __future__ import annotations

from .child_source import (
    ChildSource,
    ExplicitFileSource,
    GlobSource,
    LoadedSettingsSource,
    expand_child_sources,
)
from .orchestrator import IMultiRunOrchestrator, MultiRunOrchestrator
from .spec import MultiRunSpec, RunSpec

__all__ = [
    "ChildSource",
    "ExplicitFileSource",
    "GlobSource",
    "IMultiRunOrchestrator",
    "LoadedSettingsSource",
    "MultiRunOrchestrator",
    "MultiRunSpec",
    "RunSpec",
    "expand_child_sources",
]
