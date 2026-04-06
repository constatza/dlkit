"""RFC-compliant URL utilities using yarl for unified URL handling.

This module provides centralized URL operations following RFC 3986 (generic URIs)
and RFC 8089 (file URIs). All URL manipulation in DLKit should use these utilities
to ensure consistency and RFC compliance.

Note: For Path-to-URI conversion (file://, sqlite://), use the functions in
path_normalizers.py which handle platform-specific path canonicalization.

URI Path Interpretation (RFC 3986):
- Absolute path: starts with "/" (e.g., "/data/file.db")
- Relative path: does not start with "/" (e.g., "data/file.db")

Authority determines interpretation:
- `scheme://` introduces authority component
- Empty authority → filesystem paths (e.g., `sqlite:///path` or `file:///path`)
- Non-empty authority → host identifier (e.g., `http://host/path`)

Examples:
    SQLite URIs:
        sqlite:///data/db       → authority="" → absolute path /data/db
        sqlite:/data/db         → no authority → absolute path /data/db
        sqlite:data/db          → no authority → relative path data/db

    File URIs:
        file:///data/file       → authority="" → absolute path /data/file
        file:/data/file         → no authority → absolute path /data/file
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from yarl import URL


def parse_url(uri: str) -> URL:
    """Parse URI string into yarl URL object.

    Args:
        uri: URI string to parse

    Returns:
        Parsed yarl URL object

    Examples:
        >>> url = parse_url("sqlite:///data/mlflow.db")
        >>> url.scheme
        'sqlite'
        >>> url.path
        '/data/mlflow.db'
    """
    return URL(uri)


def get_url_path(uri: str) -> str:
    """Extract path component from URI.

    Args:
        uri: URI string

    Returns:
        Path component of the URI

    Examples:
        >>> get_url_path("sqlite:///data/mlflow.db")
        '/data/mlflow.db'

        >>> get_url_path("file:///tmp/file.txt")
        '/tmp/file.txt'

        >>> get_url_path("http://example.com/path/to/resource")
        '/path/to/resource'
    """
    url = parse_url(uri)
    return url.path


def get_url_scheme(uri: str) -> str:
    """Extract scheme from URI.

    Args:
        uri: URI string

    Returns:
        Scheme (lowercase)

    Examples:
        >>> get_url_scheme("sqlite:///data/db")
        'sqlite'

        >>> get_url_scheme("HTTP://example.com")
        'http'
    """
    url = parse_url(uri)
    return url.scheme


def normalize_url(
    uri: str,
    *,
    trailing_slash: Literal["add", "remove", "keep"] = "keep",
) -> str:
    """Normalize URI for comparison and storage.

    Args:
        uri: URI to normalize
        trailing_slash: How to handle trailing slashes

    Returns:
        Normalized URI string

    Examples:
        >>> normalize_url("HTTP://Example.Com:80/Path")
        'http://example.com/Path'

        >>> normalize_url("http://example.com/path/", trailing_slash="remove")
        'http://example.com/path'
    """
    url = parse_url(uri)

    # Handle trailing slash
    if trailing_slash == "remove" and url.path.endswith("/") and url.path != "/":
        url = url.with_path(url.path.rstrip("/"))
    elif trailing_slash == "add" and not url.path.endswith("/"):
        url = url.with_path(url.path + "/")

    return str(url)


def compare_urls(uri1: str, uri2: str) -> bool:
    """Compare two URIs for equality after normalization.

    Args:
        uri1: First URI
        uri2: Second URI

    Returns:
        True if URIs are equivalent after normalization

    Examples:
        >>> compare_urls("http://example.com/path", "HTTP://EXAMPLE.COM/path")
        True

        >>> compare_urls("http://example.com/path/", "http://example.com/path")
        False
    """
    # Normalize both URIs (remove trailing slashes for comparison)
    norm1 = normalize_url(uri1, trailing_slash="remove")
    norm2 = normalize_url(uri2, trailing_slash="remove")
    return norm1 == norm2


def is_file_scheme(uri: str) -> bool:
    """Check if URI uses a file-based scheme.

    Args:
        uri: URI to check

    Returns:
        True if scheme is file or sqlite

    Examples:
        >>> is_file_scheme("file:///data/file.txt")
        True

        >>> is_file_scheme("sqlite:///data/db.sqlite")
        True

        >>> is_file_scheme("http://example.com")
        False
    """
    scheme = get_url_scheme(uri)
    return scheme in {"file", "sqlite"}


def is_web_scheme(uri: str) -> bool:
    """Check if URI uses a web-based scheme.

    Args:
        uri: URI to check

    Returns:
        True if scheme is http or https

    Examples:
        >>> is_web_scheme("http://example.com")
        True

        >>> is_web_scheme("https://example.com")
        True

        >>> is_web_scheme("file:///data/file.txt")
        False
    """
    scheme = get_url_scheme(uri)
    return scheme in {"http", "https"}


def uri_to_path(uri: str) -> Path:
    """Convert file or sqlite URI to Path object.

    Args:
        uri: File or SQLite URI

    Returns:
        Path object from URI

    Raises:
        ValueError: If URI is not a file or sqlite scheme

    Examples:
        >>> uri_to_path("file:///data/file.txt")
        PosixPath('/data/file.txt')

        >>> uri_to_path("sqlite:///data/mlflow.db")
        PosixPath('/data/mlflow.db')
    """
    if not is_file_scheme(uri):
        msg = f"URI must be file or sqlite scheme, got: {uri}"
        raise ValueError(msg)

    path_str = get_url_path(uri)
    return Path(path_str)


def build_http_url(
    host: str,
    port: int | None = None,
    path: str = "",
    scheme: str = "http",
) -> str:
    """Build HTTP/HTTPS URL from components.

    Args:
        host: Hostname or IP address
        port: Port number (optional)
        path: Path component (optional)
        scheme: URL scheme (default: http)

    Returns:
        Complete HTTP URL

    Examples:
        >>> build_http_url("localhost", 5000)
        'http://localhost:5000'

        >>> build_http_url("example.com", 443, "/api/v1", "https")
        'https://example.com:443/api/v1'
    """
    url = URL.build(scheme=scheme, host=host, port=port, path=path)
    return str(url)
