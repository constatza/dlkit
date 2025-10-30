from __future__ import annotations

import os
from os import fspath
from pathlib import Path

from pydantic_core import Url

from dlkit.core.datatypes.urls import tilde_expand_strict


def mkdir_for_local(uri: Url | str) -> None:
    """Ensure the local directory for the given URI or path exists.

    Uses Pydantic URL types for proper parsing.

    SQLite URI semantics on Unix:
    - sqlite:///path or sqlite:///./path -> relative path
    - sqlite:////path -> absolute path /path

    Args:
        uri: Either a Pydantic Url object or a string (path or URI)
    """
    # Plain string without scheme - treat as path
    if isinstance(uri, str):
        if "://" not in uri:
            # Plain path string
            Path(uri).parent.mkdir(parents=True, exist_ok=True)
            return

        # Try to parse as URL
        try:
            uri = Url(uri)
        except Exception:
            # If parsing fails, treat as plain path
            Path(uri).parent.mkdir(parents=True, exist_ok=True)
            return

    # Now we have a Pydantic Url object
    scheme = uri.scheme
    if scheme not in ("file", "sqlite"):
        # Non-local schemes: nothing to do
        return

    path_str = uri.path or ""

    if os.name == "nt":
        # Windows: strip leading slashes
        path_str = path_str.lstrip(r"/")
    else:
        # Unix: handle SQLite URI path semantics
        # The key insight: after Pydantic parses "sqlite:///./path",
        # the path component is "/./path" (starts with /. for relative)
        # or "//path" for absolute (from sqlite:////path)

        # Collapse excessive leading slashes (5+ slashes) down to 2
        while path_str.startswith("///"):
            path_str = path_str[1:]

        # Now distinguish:
        # "//path" -> absolute path "/path" (from sqlite:////path)
        # "/path" where path starts with . or .. -> relative (from sqlite:///./path)
        # "/path" otherwise -> relative (from sqlite:///path)
        if path_str.startswith("//"):
            # Absolute: strip one slash to get /path
            path_str = path_str[1:]
        elif path_str.startswith("/"):
            # Relative: strip leading slash
            path_str = path_str[1:]

    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


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
