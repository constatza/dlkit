"""Universal tilde expansion utility for URLs and paths.

This module provides tilde expansion that works before any Pydantic validation,
ensuring consistent behavior across all URL and path types.

Guideline: Avoid importing `urllib`, `requests`, `httpx`, or other third-party
URL wrappers here. All parsing and rebuilding is handled via `pydantic_core.Url`
helpers to guarantee identical semantics to the validators that consume them.
"""

import re
from pathlib import Path
from typing import Any

from pydantic_core import Url

# RFC 3986 scheme: ALPHA *( ALPHA / DIGIT / "+" / "-" / "." )
_URL_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://")


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
    # Guard: Non-strings pass through unchanged
    if not isinstance(value, str):
        return value

    # Guard: Strings without tilde pass through unchanged
    if "~" not in value:
        return value

    return _expand_tilde_in_string(value)


def _expand_tilde_in_string(text: str) -> str:
    # Guard: No tilde, return unchanged
    if "~" not in text:
        return text

    home = _home()

    # Route based on URL vs plain path.
    # Use strict RFC 3986 scheme check so paths like "~/://" aren't misclassified.
    if _URL_SCHEME_RE.match(text):
        return _expand_tilde_in_url(text, home)

    return _expand_tilde_in_path(text, home)


def _expand_tilde_in_url(url: str, home: str) -> str:
    # Guard: Must be URL format
    if "://" not in url:
        raise ValueError("Tilde must appear at the start of the path")

    # Guard: Must parse as valid URL
    try:
        parsed = Url(url)
    except Exception:
        return url

    # Extract original netloc for port detection
    original_netloc = _extract_netloc(url)
    include_port = _netloc_has_explicit_port(original_netloc)

    # Match scheme to dispatch to specialized handlers
    match parsed.scheme:
        case "file":
            return _expand_file_url_tilde(parsed, home, url, include_port)
        case _:
            return _expand_generic_url_tilde(parsed, home, url, include_port)


def _expand_file_url_tilde(parsed: Url, home: str, url: str, include_port: bool) -> str:
    """Handle tilde expansion for file:// URLs.

    Args:
        parsed: Parsed URL object
        home: Home directory path
        url: Original URL string
        include_port: Whether to include port in rebuilt URL

    Returns:
        Expanded URL string
    """
    # Guard: Only expand if host is None or '~' AND no auth credentials
    if not (parsed.host in {None, "~"} and not parsed.username and not parsed.password):
        return _expand_generic_url_tilde(parsed, home, url, include_port)

    # Combine host and path for file:// special handling
    combined = (parsed.host or "") + (parsed.path or "")

    # Guard: Must have tilde in combined path
    if "~" not in combined:
        return url

    # Expand tilde in file:// URLs - only handle valid ~/path pattern
    if combined.startswith("~/"):
        # For file:// URLs, strip leading / from home since file:/// provides the root
        expanded = home.lstrip("/") + combined[1:]
    else:
        # Any other tilde usage - just replace and let it fail if wrong
        expanded = combined.replace("~", home.lstrip("/"))

    # Normalize URL path (ensures leading slash, converts backslashes)
    new_path = _normalize_url_path(expanded)

    return _rebuild_url(
        parsed,
        path=new_path,
        host_override="",
        include_port=include_port,
    )


def _expand_generic_url_tilde(parsed: Url, home: str, url: str, include_port: bool) -> str:
    """Handle tilde expansion for non-file URLs (http, sqlite, etc).

    Args:
        parsed: Parsed URL object
        home: Home directory path
        url: Original URL string
        include_port: Whether to include port in rebuilt URL

    Returns:
        Expanded URL string
    """
    path = parsed.path or ""

    # Guard: No tilde in path, return unchanged
    if "~" not in path:
        return url

    # Simple replacement for generic URLs
    new_path = path.replace("~", home)

    # Normalize only if absolute path
    new_path = (
        _normalize_url_path(new_path) if path.startswith("/") else new_path.replace("\\", "/")
    )

    return _rebuild_url(parsed, path=new_path, include_port=include_port)


def _expand_tilde_in_path(path: str, home: str) -> str:
    """Expand tilde in file system paths.

    Only handles valid tilde patterns:
    - `~` → home directory
    - `~/path` → home/path

    Everything else passes through unchanged and will fail naturally.
    """
    match path:
        case "~":
            return home
        case s if s.startswith("~/"):
            return _normalize_file_path(home + s[1:])
        case _:
            return path


def _home() -> str:
    return _normalize_file_path(str(Path.home()))


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

    normalized_path = _normalize_url_path(path) if path.startswith("/") else path.replace("\\", "/")
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


def _normalize_url_path(path: str) -> str:
    """Normalize path component for URL reconstruction (RFC 3986).

    Always ensures leading slash and forward slashes for URL path components.
    This is used when building URLs like file:///path or sqlite:///path.

    Args:
        path: URL path component to normalize

    Returns:
        Normalized path with leading slash and forward slashes
    """
    normalized = path.replace("\\", "/")

    # URL path components always need leading slash
    if not normalized.startswith("/"):
        normalized = "/" + normalized.lstrip("/")

    # Collapse triple slashes to single
    if normalized.startswith("///"):
        normalized = "/" + normalized.lstrip("/")

    return normalized


def _normalize_file_path(path: str) -> str:
    """Normalize file system path using pathlib for cross-platform support.

    Handles platform-specific absolute paths correctly:
    - Unix: /home/user/file
    - Windows: C:/Users/user/file

    Args:
        path: File system path to normalize

    Returns:
        Normalized path as string with forward slashes (pathlib format)
    """
    from pathlib import PurePath

    # Use pathlib for proper cross-platform handling
    p = PurePath(path)

    # Convert to forward slashes (pathlib's portable format)
    return str(p).replace("\\", "/")


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
