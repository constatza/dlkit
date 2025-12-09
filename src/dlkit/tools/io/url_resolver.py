"""Centralized local URI resolution and normalization using yarl.

This module provides a single source of truth for handling local URIs
(`file` and `sqlite`) plus plain filesystem paths. It leverages `yarl`
for RFC-compliant parsing/building and defers platform-aware path
canonicalization to the existing helpers in `path_normalizers`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from yarl import URL

from dlkit.tools.io.path_normalizers import canonicalize_file_path

LocalScheme = Literal["file", "sqlite"]


def scheme(uri: str) -> str:
    """Return lower-cased scheme or empty string for plain paths."""
    url = _parse(uri)
    return url.scheme if url is not None else ""


def is_local_uri(uri: str) -> bool:
    """Return True when scheme is file or sqlite."""
    sch = scheme(uri)
    return sch in {"file", "sqlite"}


def resolve_local_uri(uri: str, root: Path) -> Path:
    """Resolve file/sqlite/plain path to an absolute Path using root for relatives."""
    url = _parse(uri)
    if url is None or not url.scheme:
        return _resolve_relative(uri, root)

    match url.scheme:
        case "file":
            path_str = _file_path_from_url(url)
        case "sqlite":
            path_str = _sqlite_path_from_url(url)
        case _:
            raise ValueError(f"Unsupported scheme for local resolution: {url.scheme}")

    return _resolve_relative(path_str, root)


def normalize_uri(uri: str, root: Path) -> str:
    """Normalize a URI string (file/sqlite/plain path) to a canonical string form."""
    url = _parse(uri)
    if url is None or not url.scheme:
        # Plain path -> normalized file:// absolute URI
        resolved = _resolve_relative(uri, root)
        return _file_uri(resolved)

    match url.scheme:
        case "file":
            resolved = _resolve_relative(_file_path_from_url(url), root)
            return _file_uri(resolved)
        case "sqlite":
            resolved = _resolve_relative(_sqlite_path_from_url(url), root)
            return _sqlite_uri(resolved)
        case _:
            raise ValueError(f"Unsupported scheme for normalization: {url.scheme}")


def build_uri(path: Path, *, scheme: LocalScheme) -> str:
    """Build a normalized URI string from a Path."""
    if scheme == "file":
        return _file_uri(path)
    if scheme == "sqlite":
        return _sqlite_uri(path)
    raise ValueError(f"Unsupported scheme: {scheme}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse(candidate: str) -> URL | None:
    try:
        return URL(candidate)
    except Exception:
        return None


def _canonical_path(path_str: str) -> str:
    return canonicalize_file_path(path_str)


def _resolve_relative(path_str: str | Path, root: Path) -> Path:
    path = Path(path_str) if isinstance(path_str, str) else path_str
    canonical = _canonical_path(path.as_posix())
    candidate = Path(canonical)
    if candidate.is_absolute():
        # Disallow embedded parent/self segments in absolute paths
        if any(part in {"..", "."} for part in candidate.parts):
            raise ValueError(f"Invalid absolute path containing traversal segments: {candidate}")
        return candidate.resolve()
    return (root / candidate).resolve()


def _file_path_from_url(url: URL) -> str:
    # RFC8089: file URLs may have empty host; yarl gives .path sans authority
    path = url.path or ""
    if path.startswith("//") and not path.startswith("///"):
        return f"/{path.lstrip('/')}"
    if path.startswith("/"):
        return path
    return _strip_leading_dot_segments(path)


def _sqlite_path_from_url(url: URL) -> str:
    # SQLAlchemy dialect expects 4 slashes for absolute, 3 for relative.
    # yarl parses sqlite:////abs -> path="//abs"; sqlite:///rel -> path="/rel"
    path = url.path or ""
    if path.startswith("//") and not path.startswith("///"):
        return path[1:]
    if path.startswith("/"):
        return _strip_leading_dot_segments(path[1:])
    return _strip_leading_dot_segments(path)


def _sqlite_uri(path: Path) -> str:
    canonical = _canonical_path(path.as_posix())
    if canonical.startswith("/"):
        return f"sqlite:////{canonical.lstrip('/')}"
    return f"sqlite:///{canonical}"


def _file_uri(path: Path) -> str:
    resolved = path if path.is_absolute() else path.resolve()
    canonical = _canonical_path(resolved.as_posix())
    # RFC8089 absolute form
    if canonical.startswith("/"):
        return f"file://{canonical}"
    return f"file:///{canonical}"


def _strip_leading_dot_segments(path_str: str) -> str:
    # Remove leading ./ or ../ segments for relative paths to prevent escaping root
    while path_str.startswith("./"):
        path_str = path_str[2:]
    while path_str.startswith("../"):
        path_str = path_str[3:]
    return path_str
