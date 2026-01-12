from __future__ import annotations

import os
from os import fspath
from pathlib import Path

from pydantic_core import Url

from dlkit.core.datatypes.urls import tilde_expand_strict


def mkdir_for_local(uri: Url | str, *, root: Path | None = None) -> None:
    """Ensure the local directory for the given URI or path exists."""
    # Lazy import to avoid circular import during package initialization
    from dlkit.tools.io import url_resolver

    root_path = root or Path.cwd()
    uri_str = str(uri)

    # Guard: Ignore remote URIs
    if "://" in uri_str and not url_resolver.is_local_uri(uri_str):
        return

    # Try to resolve URI, fallback to plain path on error
    try:
        resolved = url_resolver.resolve_local_uri(uri_str, root_path)
    except ValueError:
        # Unsupported scheme or malformed URI: treat as plain path parent creation
        Path(uri_str).parent.mkdir(parents=True, exist_ok=True)
        return

    # Determine target directory based on scheme
    current_scheme = url_resolver.scheme(uri_str)
    match current_scheme:
        case "sqlite":
            target_dir = resolved.parent
        case _:
            # Heuristic: create the path itself when it looks like a directory
            target_dir = resolved if resolved.suffix == "" else resolved.parent

    target_dir.mkdir(parents=True, exist_ok=True)


def normalize_user_path(value: str | Path | None, *, require_absolute: bool = False) -> Path | None:
    """Normalize user-supplied paths from CLI/config overrides.

    - Expands ``~`` to home directory
    - Resolves relative paths against current working directory
    - Optionally enforces absolute paths (returns ``None`` if not achievable)
    - Does NOT fix user mistakes - bad paths will fail naturally with clear errors
    """
    # Guard: None input
    if value is None:
        return None

    # Expand tilde
    if isinstance(value, Path):
        candidate = value.expanduser()
    else:
        normalized = tilde_expand_strict(fspath(value))
        candidate = Path(normalized).expanduser()

    # Guard: Already absolute
    if candidate.is_absolute():
        return candidate.resolve()

    # Guard: Require absolute but not absolute
    if require_absolute:
        return None

    # Make absolute relative to cwd
    return (Path.cwd() / candidate).resolve()


def coerce_root_dir_to_absolute(value: str | Path | None) -> Path | None:
    """Convert SESSION.root_dir style values into absolute Paths when possible."""
    return normalize_user_path(value, require_absolute=True)


def recommended_uvicorn_workers() -> int:
    """Compute worker count using the standard (2 * cores) + 1 heuristic."""
    num_cores = os.cpu_count() or 8
    # Gunicorn’s recommended formula: (2 * n) + 1
    return (2 * num_cores) + 1
