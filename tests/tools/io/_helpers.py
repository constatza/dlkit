"""Helper utilities for URL and path resolution testing.

This module provides reusable utilities for testing URL handling,
path resolution, and configuration scenarios across the test suite.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

# URL test dataflow constants
SQLITE_URLS = [
    "sqlite:///path/to/database.db",
    "sqlite:///absolute/path/mlflow.db",
    "sqlite:///~/home/user/project/mlflow.db",
    "sqlite://user:pass@host:5432/database",
    "sqlite:////absolute/path/with/four/slashes.db",
]

NON_FILE_URL_SCHEMES = [
    "sqlite",
    "postgresql",
    "mysql",
    "redis",
    "mongodb",
    "http",
    "https",
    "ftp",
    "sftp",
    "s3",
    "gs",
    "hdfs",
]

FILE_URLS = [
    "file:///absolute/path/to/file.txt",
    "file://relative/path/to/file.txt",
    "file:///~/home/user/file.txt",
]

MALFORMED_URLS = [
    "://missing-scheme",
    "scheme-without-colon-slash",
    "http:/missing-slash",
    "sqlite:///path\x00with\x00nulls.db",
    "",  # Empty string
    "sqlite://malformed?url=structure",
]


def create_test_urls_by_scheme() -> dict[str, list[str]]:
    """Create a mapping of URL schemes to example URLs for testing.

    Returns:
        Dictionary mapping scheme names to lists of example URLs.
    """
    return {
        "sqlite": [
            "sqlite:///local.db",
            "sqlite:///path/to/db.sqlite",
            "sqlite:///~/user/db.db",
            "sqlite://user:pass@host:5432/db",
        ],
        "postgresql": [
            "postgresql://user:pass@localhost:5432/dbname",
            "postgresql://localhost/dbname",
        ],
        "mysql": [
            "mysql://root:password@localhost:3306/database",
            "mysql://localhost/database",
        ],
        "redis": [
            "redis://localhost:6379/0",
            "redis://user:pass@redis.example.com:6379",
        ],
        "http": [
            "http://example.com/api",
            "http://localhost:8000/data",
        ],
        "https": [
            "https://secure.example.com/api",
            "https://api.service.com/v1/data",
        ],
        "s3": [
            "s3://bucket-name/path/to/object",
            "s3://my-dataflow-bucket/datasets/train.csv",
        ],
        "file": [
            "file:///absolute/path/file.txt",
            "file://relative/path/file.txt",
        ],
    }


def create_mock_settings_with_urls(root_path: Path, urls: dict[str, str]) -> Mock:
    """Create a mock settings object with specified URLs for testing.

    Args:
        root_path: Root directory path for the mock settings
        urls: Dictionary of section.field -> URL mappings

    Returns:
        Mock settings object with the specified URL configuration.
    """
    mock_paths = Mock()
    mock_paths.root_dir = str(root_path)
    mock_paths.data_dir = "dataflow"
    mock_paths.output_dir = "outputs"
    mock_paths.model_copy = Mock(return_value=mock_paths)

    mock_mlflow = Mock()
    mock_mlflow.enabled = True
    mock_mlflow.experiment_name = "test_experiment"
    mock_mlflow.run_name = None
    mock_mlflow.register_model = True
    mock_mlflow.model_copy = Mock(return_value=mock_mlflow)

    mock_settings = Mock()
    mock_settings.MLFLOW = mock_mlflow
    mock_settings.DATASET = None
    mock_settings.MODEL = None
    mock_settings.TRAINING = None
    mock_settings.model_copy = Mock(return_value=mock_settings)

    return mock_settings


def create_config_content_with_urls(
    urls: dict[str, str], include_regular_paths: bool = True
) -> str:
    """Create TOML configuration content with specified URLs.

    Args:
        urls: Dictionary mapping config keys to URL values
        include_regular_paths: Whether to include regular file paths in the config

    Returns:
        TOML configuration string.
    """
    config_lines = []

    # PATHS section - deprecated and removed

    # MLFLOW section
    if any(key.startswith("mlflow") for key in urls):
        config_lines.extend([
            "[MLFLOW]",
            "enabled = true",
            'experiment_name = "test_experiment"',
        ])

        config_lines.append("")

    # Custom URL sections
    custom_urls = {k: v for k, v in urls.items() if not k.startswith("mlflow")}
    if custom_urls:
        config_lines.extend([
            "[CUSTOM_URLS]",
        ])
        for key, url in custom_urls.items():
            config_lines.append(f'{key} = "{url}"')
        config_lines.append("")

    # SESSION section
    if include_regular_paths:
        config_lines.extend([
            "[SESSION]",
            'name = "test_session"',
        ])

    return "\n".join(config_lines)


def assert_url_preserved(original: str, resolved: str, scheme: str) -> None:
    """Assert that a URL with the given scheme was preserved correctly.

    Args:
        original: Original URL value
        resolved: Resolved URL value
        scheme: Expected URL scheme

    Raises:
        AssertionError: If URL was not preserved correctly
    """
    assert resolved.startswith(f"{scheme}://"), (
        f"Scheme should be preserved: {original} -> {resolved}"
    )

    if scheme != "file":
        # Non-file URLs should be completely unchanged
        assert resolved == original, f"Non-file URL should be unchanged: {original} -> {resolved}"
    else:
        # File URLs may have path resolution but should maintain scheme
        assert resolved.startswith("file://"), f"File URL scheme should be preserved: {resolved}"


def assert_regular_path_resolved(original: str, resolved: str, expected_root: Path) -> None:
    """Assert that a regular (non-URL) path was resolved correctly.

    Args:
        original: Original path value
        resolved: Resolved path value
        expected_root: Expected root directory for resolution

    Raises:
        AssertionError: If path was not resolved correctly
    """
    if original.startswith("/"):
        # Absolute path - should be resolved but location unchanged
        assert Path(resolved).is_absolute(), (
            f"Absolute path should remain absolute: {original} -> {resolved}"
        )
    elif original.startswith("~/"):
        # Tilde path - should expand to home directory
        expected = Path.home() / original[2:]
        assert Path(resolved) == expected.resolve(), (
            f"Tilde path should expand: {original} -> {resolved}"
        )
    else:
        # Relative path - should resolve against root
        expected = (expected_root / original).resolve()
        assert Path(resolved) == expected, (
            f"Relative path should resolve against root: {original} -> {resolved}"
        )


def create_temporary_config_file(
    content: str, tmp_path: Path, filename: str = "test_config.toml"
) -> Path:
    """Create a temporary configuration file with the given content.

    Args:
        content: TOML configuration content
        tmp_path: Temporary directory path
        filename: Name for the config file

    Returns:
        Path to the created config file.
    """
    config_file = tmp_path / filename
    config_file.write_text(content)
    return config_file


def extract_url_components(url: str) -> dict[str, str]:
    """Extract components from a URL for testing purposes.

    Args:
        url: URL to parse

    Returns:
        Dictionary with URL components (scheme, netloc, path, etc.)
        Returns empty dict if URL is malformed.
    """
    try:
        if "://" not in url:
            return {}

        scheme, rest = url.split("://", 1)
        components = {"scheme": scheme.lower()}

        if "/" in rest:
            netloc, path = rest.split("/", 1)
            components["netloc"] = netloc
            components["path"] = "/" + path
        else:
            components["netloc"] = rest
            components["path"] = ""

        return components
    except Exception:
        return {}


def is_valid_url(url: str) -> bool:
    """Check if a string represents a valid URL format.

    Args:
        url: String to check

    Returns:
        True if the string has valid URL format, False otherwise.
    """
    if not isinstance(url, str) or not url:
        return False

    if "://" not in url:
        return False

    try:
        scheme, rest = url.split("://", 1)
        return len(scheme) > 0 and len(rest) > 0
    except Exception:
        return False


def get_url_scheme(url: str) -> str | None:
    """Extract the scheme from a URL.

    Args:
        url: URL string

    Returns:
        Scheme name (lowercase) or None if not a valid URL.
    """
    if not is_valid_url(url):
        return None

    return url.split("://", maxsplit=1)[0].lower()
