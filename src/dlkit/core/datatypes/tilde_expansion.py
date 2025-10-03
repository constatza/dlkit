"""Universal tilde expansion utility for URLs and paths.

This module provides tilde expansion that works before any Pydantic validation,
ensuring consistent behavior across all URL and path types.

Guideline: Avoid importing `urllib`, `requests`, `httpx`, or other third-party
URL wrappers here. All parsing and rebuilding is handled via `pydantic_core.Url`
helpers to guarantee identical semantics to the validators that consume them.
"""

from pathlib import Path
from typing import Any

from pydantic_core import Url


def expand_tilde_in_value(value: Any) -> Any:
    """Expand tilde (~) in string values before any other validation.

    This function handles tilde expansion in:
    - Plain file paths: "~/project/file.txt"
    - URLs with paths: "sqlite:///~/project/db.sqlite", "file:///~/docs/file.txt"

    Plain paths that contain '~' outside the leading segment are rejected with a
    ValueError. URL paths are exempt from this restriction.

    Args:
        value: Input value (any type, only strings are processed)

    Returns:
        Value with tilde expanded if it was a string containing ~, otherwise unchanged
    """
    if not isinstance(value, str) or "~" not in value:
        return value

    return _expand_tilde_in_string(value)


def _expand_tilde_in_string(text: str) -> str:
    if "~" not in text:
        return text

    home = _home()

    if "://" in text:
        return _expand_tilde_in_url(text, home)

    return _expand_tilde_in_path(text, home)


def _expand_tilde_in_url(url: str, home: str) -> str:
    if "://" not in url:
        raise ValueError("Tilde must appear at the start of the path")

    try:
        parsed = Url(url)
    except Exception:
        return url

    original_netloc = _extract_netloc(url)
    include_port = _netloc_has_explicit_port(original_netloc)

    if (
        parsed.scheme == "file"
        and parsed.host in {None, "~"}
        and not parsed.username
        and not parsed.password
    ):
        combined = (parsed.host or "") + (parsed.path or "")
        if combined.startswith("~/"):
            expanded = home + combined[1:]
        elif combined.startswith("/~/"):
            expanded = "/" + home.lstrip("/") + combined[2:]
        elif "~" in combined:
            expanded = combined.replace("~", home)
        else:
            return url

        new_path = expanded if expanded.startswith("/") else "/" + expanded
        new_path = _normalize_path(new_path, ensure_leading_slash=True)
        return _rebuild_url(
            parsed,
            path=new_path,
            host_override="",
            include_port=include_port,
        )

    path = parsed.path or ""

    if "~" not in path:
        return url

    new_path = path.replace("~", home)
    new_path = _normalize_path(new_path, ensure_leading_slash=path.startswith("/"))
    return _rebuild_url(parsed, path=new_path, include_port=include_port)


def _expand_tilde_in_path(path: str, home: str) -> str:
    if path == "~":
        return home

    if path.startswith("~/"):
        if "~" in path[2:]:
            raise ValueError("Tilde may only appear at the start of the path")
        return _normalize_path(home + path[1:], ensure_leading_slash=True)

    if path.startswith("/~"):
        if "~" in path[2:]:
            raise ValueError("Tilde may only appear at the start of the path")
        combined = "/" + home.lstrip("/") + path[2:]
        return _normalize_path(combined, ensure_leading_slash=True)

    if path.startswith("~"):
        if "~" in path[1:]:
            raise ValueError("Tilde may only appear at the start of the path")
        return path

    if "~" in path:
        raise ValueError("Tilde must appear at the start of the path")

    return path


def _home() -> str:
    return _normalize_path(str(Path.home().expanduser()), ensure_leading_slash=True)


def _rebuild_url(
    parsed: Url,
    *,
    path: str,
    host_override: str | None = None,
    include_port: bool | None = None,
) -> str:
    netloc = _render_netloc(
        parsed,
        host_override=host_override,
        include_port=include_port,
    )

    result = f"{parsed.scheme}:"

    if netloc or parsed.scheme in {"file", "sqlite"} or path.startswith("/"):
        result += "//" + netloc

    normalized_path = _normalize_path(path, ensure_leading_slash=path.startswith("/"))
    result += normalized_path

    if parsed.query:
        result += f"?{parsed.query}"

    if parsed.fragment:
        result += f"#{parsed.fragment}"

    return result


def _render_netloc(
    parsed: Url,
    *,
    host_override: str | None,
    include_port: bool | None,
) -> str:
    username = parsed.username or ""
    password = parsed.password

    host = parsed.host if host_override is None else host_override
    host = host or ""

    netloc = ""

    if username:
        netloc += username
        if password:
            netloc += f":{password}"
        netloc += "@"

    netloc += host

    allow_port = include_port if include_port is not None else True

    if allow_port and parsed.port is not None:
        netloc += f":{parsed.port}"

    return netloc


def _normalize_path(value: str, *, ensure_leading_slash: bool = False) -> str:
    normalized = value.replace("\\", "/")

    if ensure_leading_slash and not normalized.startswith("/"):
        normalized = "/" + normalized.lstrip("/")

    if ensure_leading_slash and normalized.startswith("///"):
        # Collapse to a single leading slash when accidental triples occur
        normalized = "/" + normalized.lstrip("/")

    return normalized


def _extract_netloc(url: str) -> str:
    scheme_sep = url.find("://")
    if scheme_sep == -1:
        return ""

    remainder = url[scheme_sep + 3 :]
    end = len(remainder)
    for idx, ch in enumerate(remainder):
        if ch in "/?#":
            end = idx
            break

    return remainder[:end]


def _netloc_has_explicit_port(netloc: str) -> bool:
    if not netloc:
        return False

    authority = netloc
    if "@" in authority:
        authority = authority.rsplit("@", 1)[1]

    if authority.startswith("["):
        closing = authority.find("]")
        if closing == -1:
            return False
        return len(authority) > closing + 1 and authority[closing + 1] == ":"

    return ":" in authority


def create_tilde_expanding_validator(base_validator_func):
    """Decorator to add tilde expansion to any validator function.

    Args:
        base_validator_func: Function that validates the value after tilde expansion

    Returns:
        New validator function that expands tildes first, then validates

    Example:
        @create_tilde_expanding_validator
        def validate_sqlite_url(value: str) -> SQLiteUri:
            # value will already have tildes expanded
            return SQLiteUri(value)
    """

    def validator_with_tilde_expansion(value: Any):
        # Expand tildes first
        expanded_value = expand_tilde_in_value(value)
        # Then validate with original validator
        return base_validator_func(expanded_value)

    return validator_with_tilde_expansion
