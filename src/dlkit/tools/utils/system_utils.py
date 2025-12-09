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

    # Ignore clearly remote schemes early
    if "://" in uri_str and not url_resolver.is_local_uri(uri_str):
        return

    try:
        resolved = url_resolver.resolve_local_uri(uri_str, root_path)
    except ValueError:
        # Unsupported scheme or malformed URI: treat as plain path parent creation
        Path(uri_str).parent.mkdir(parents=True, exist_ok=True)
        return

    current_scheme = url_resolver.scheme(uri_str)
    if current_scheme == "sqlite":
        target_dir = resolved.parent
    else:
        # Heuristic: create the path itself when it looks like a directory
        target_dir = resolved if resolved.suffix == "" else resolved.parent

    target_dir.mkdir(parents=True, exist_ok=True)


def _get_reference_home() -> Path:
    """Return a stable reference for the user's home directory."""
    env_home = os.environ.get("HOME") or os.environ.get("USERPROFILE")
    if env_home:
        try:
            return Path(env_home).expanduser()
        except Exception:
            pass
    return Path.home()


def _maybe_fix_missing_leading_slash(path: Path) -> Path | None:
    """Detect common absolute paths missing a leading slash and repair them."""
    reference_home = _get_reference_home()
    home_parts = tuple(part for part in reference_home.parts if part and part != reference_home.anchor)
    path_parts = path.parts

    if not home_parts:
        return None

    if len(path_parts) >= len(home_parts) and path_parts[: len(home_parts)] == home_parts:
        anchor = reference_home.anchor or os.sep
        return (Path(anchor) / Path(*path_parts)).resolve()

    return None


def normalize_user_path(value: str | Path | None, *, require_absolute: bool = False) -> Path | None:
    """Normalize user-supplied paths from CLI/config overrides.

    - Expands ``~`` using shared tilde expansion rules
    - Preserves ``Path`` instances when possible
    - Resolves relative paths against the current working directory
    - Optionally enforces absolute paths (returning ``None`` when not achievable)
    - Repairs paths missing a leading slash but clearly pointing to HOME
    """
    if value is None:
        return None

    if isinstance(value, Path):
        candidate = value.expanduser()
    else:
        normalized = tilde_expand_strict(fspath(value))
        candidate = Path(normalized).expanduser()

    repaired = _maybe_fix_missing_leading_slash(candidate)
    if repaired is not None:
        candidate = repaired

    if candidate.is_absolute():
        return candidate.resolve()

    if require_absolute:
        return None

    return (Path.cwd() / candidate).resolve()


def coerce_root_dir_to_absolute(value: str | Path | None) -> Path | None:
    """Convert SESSION.root_dir style values into absolute Paths when possible."""
    return normalize_user_path(value, require_absolute=True)


def recommended_uvicorn_workers() -> int:
    """Compute worker count using the standard (2 * cores) + 1 heuristic."""
    num_cores = os.cpu_count() or 8
    # Gunicorn’s recommended formula: (2 * n) + 1
    return (2 * num_cores) + 1
