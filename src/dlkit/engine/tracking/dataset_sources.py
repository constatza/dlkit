"""Dataset source discovery and fingerprinting for lineage logging.

Everything here is config-driven and never reads dataset content:
- ``iter_entries`` / ``path_entries`` are the single view of the configured
  Feature/Target entries (and of which of them are file-backed).
- ``DatasetSourceCollector`` turns settings into an ordered list of source paths.
- ``compute_sources_fingerprint`` digests those paths plus cheap file metadata.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from dlkit.engine.data.families import resolve_family
from dlkit.infrastructure.config.entry_factories import AnyEntry
from dlkit.infrastructure.config.entry_types import PathBasedEntry
from dlkit.infrastructure.config.enums import DatasetFamily
from dlkit.infrastructure.config.job_config import JobConfig

type _WorkflowSettings = JobConfig


def _append_unique_path(paths: list[str], candidate: Any) -> None:
    """Append candidate to paths if it is a non-empty path-like value."""
    if not isinstance(candidate, (str, Path)):
        return
    value = str(candidate).strip()
    if value and value not in paths:
        paths.append(value)


def iter_entries(settings: _WorkflowSettings) -> tuple[AnyEntry, ...]:
    """Return configured features followed by targets (empty without data settings)."""
    ds_settings = settings.data
    if ds_settings is None:
        return ()
    features = getattr(ds_settings, "features", None) or ()
    targets = getattr(ds_settings, "targets", None) or ()
    return (*features, *targets)


def path_entries(settings: _WorkflowSettings) -> tuple[PathBasedEntry, ...]:
    """Return file-backed entries whose path is resolved (placeholders excluded)."""
    return tuple(
        entry
        for entry in iter_entries(settings)
        if isinstance(entry, PathBasedEntry) and not entry.is_placeholder()
    )


def _source_stamp(source: str) -> tuple[str, int, int] | tuple[str]:
    """Return ``(source, size, mtime_ns)`` for an existing local path, else ``(source,)``.

    Metadata only, never file content. Remote URIs and missing files have no local
    ``stat`` and degrade to path-only instead of raising.
    """
    try:
        stat = Path(source).stat()
    except OSError:
        return (source,)
    return (source, stat.st_size, stat.st_mtime_ns)


def compute_sources_fingerprint(sources: list[str] | tuple[str, ...]) -> str:
    """Content-independent digest of source paths plus their size and mtime.

    Order-independent. Detects a rewritten file at an unchanged path without reading it.
    """
    payload = json.dumps(sorted(_source_stamp(source) for source in sources), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@runtime_checkable
class ISourceExtractionStrategy(Protocol):
    """Config-only path extraction strategy."""

    def collect(self, settings: _WorkflowSettings) -> list[str]:
        """Return ordered unique source paths."""
        ...


class GraphConfigSourceStrategy:
    """Extract graph source paths from graph dataset config fields."""

    _FIELDS = ("x", "edge_index", "y")

    def collect(self, settings: _WorkflowSettings) -> list[str]:
        sources: list[str] = []
        ds_settings = settings.data
        if ds_settings is None:
            return sources
        for field_name in self._FIELDS:
            _append_unique_path(sources, getattr(ds_settings, field_name, None))
        return sources


class CustomConfigSourceStrategy:
    """Extract custom lineage source list from DATASET.source_paths."""

    def collect(self, settings: _WorkflowSettings) -> list[str]:
        sources: list[str] = []
        ds_settings = settings.data
        if ds_settings is None:
            return sources

        configured = getattr(ds_settings, "source_paths", None)
        if not isinstance(configured, (list, tuple, set)):
            _append_unique_path(sources, configured)
            return sources

        for value in configured:
            _append_unique_path(sources, value)
        return sources


class DatasetSourceCollector:
    """Collect canonical dataset sources from settings using family-specific strategies."""

    def __init__(self) -> None:
        self._graph = GraphConfigSourceStrategy()
        self._custom = CustomConfigSourceStrategy()

    def collect(self, settings: _WorkflowSettings) -> list[str]:
        sources: list[str] = []
        for entry in path_entries(settings):
            _append_unique_path(sources, entry.path)

        match resolve_family(settings):
            case DatasetFamily.GRAPH:
                extras = self._graph.collect(settings)
            case _:
                extras = self._custom.collect(settings)

        for value in extras:
            _append_unique_path(sources, value)
        return sources
