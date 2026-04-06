"""Shared helpers for normalizing local filesystem paths and URIs."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath, PureWindowsPath

from pydantic import TypeAdapter, ValidationError

from dlkit.infrastructure.types.urls import FileUrl

_FILE_URL_ADAPTER = TypeAdapter(FileUrl)


def canonicalize_file_path(path_str: str) -> str:
    """Return a POSIX-formatted path with consistent drive casing."""
    if not path_str:
        return path_str

    candidate = path_str.replace("\\", "/")

    if (
        candidate.startswith("//")
        and len(candidate) > 3
        and candidate[2].isalpha()
        and candidate[3] == ":"
    ):
        candidate = candidate[2:]

    if (
        candidate.startswith("/")
        and len(candidate) > 2
        and candidate[1].isalpha()
        and candidate[2] == ":"
    ):
        candidate = candidate[1:]

    windows_path = PureWindowsPath(candidate)
    if windows_path.drive:
        posix = windows_path.as_posix()
        drive = windows_path.drive.upper()
        return f"{drive}{posix[len(windows_path.drive) :]}"

    return PurePosixPath(candidate).as_posix()


def canonicalize_file_uri(uri: str) -> str:
    """Canonicalize a file:// URI for reliable comparison."""
    stripped = uri.rstrip("/")

    try:
        url = _FILE_URL_ADAPTER.validate_python(stripped)
    except ValidationError:
        return stripped

    canonical_path = canonicalize_file_path(url.path)
    return f"file://{_path_component_for_file_uri(canonical_path)}"


def normalize_file_uri(uri: str, root_dir: Path) -> str | None:
    """Normalize file:// URIs relative to a root directory."""
    try:
        url = _FILE_URL_ADAPTER.validate_python(uri)
    except ValidationError:
        return None

    canonical_path = canonicalize_file_path(url.path)
    resolved_path_str = _resolve_canonical_path_str(canonical_path, root_dir)
    return f"file://{_path_component_for_file_uri(resolved_path_str)}"


def resolve_local_path(path_like: str | Path, root_dir: Path) -> Path:
    """Resolve a local path value relative to a root directory."""
    if isinstance(path_like, Path):
        candidate = path_like
        canonical = canonicalize_file_path(candidate.as_posix())
    else:
        canonical = canonicalize_file_path(str(path_like))
        candidate = Path(canonical)

    if candidate.is_absolute():
        return candidate.resolve()

    if _is_windows_absolute_string(canonical):
        return Path(canonical)

    return (root_dir / candidate).resolve()


def path_to_file_uri(path: Path) -> str:
    """Convert a local Path to a canonical file:// URI."""
    resolved = path if path.is_absolute() else path.resolve()
    canonical_path = canonicalize_file_path(resolved.as_posix())
    return f"file://{_path_component_for_file_uri(canonical_path)}"


def _path_component_for_file_uri(canonical_path: str) -> str:
    if canonical_path.startswith("/") or not canonical_path:
        return canonical_path
    if _is_windows_absolute_string(canonical_path):
        return f"/{canonical_path}"
    return canonical_path


def _is_windows_absolute_string(value: str) -> bool:
    return PureWindowsPath(value).is_absolute()


def _resolve_canonical_path_str(canonical_path: str, root_dir: Path) -> str:
    if _is_windows_absolute_string(canonical_path):
        if os.name == "nt":
            return Path(canonical_path).resolve().as_posix()
        return canonical_path

    posix_path = PurePosixPath(canonical_path)
    if posix_path.is_absolute():
        return Path(posix_path).resolve().as_posix()

    return (root_dir / Path(canonical_path)).resolve().as_posix()
