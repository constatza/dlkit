"""Shared fixtures for datatypes tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest


@pytest.fixture
def mock_home_path() -> str:
    """Mock home directory path for consistent testing.

    Returns:
        str: Mock home directory path
    """
    return "/mock/home/user"


@pytest.fixture
def sample_paths() -> dict[str, str]:
    """Sample path strings for tilde expansion testing.

    Returns:
        dict[str, str]: Collection of test paths with expected expansions
    """
    return {
        # Basic tilde patterns
        "tilde_start": "~/documents/file.txt",
        "tilde_root": "/~/documents/file.txt",
        "tilde_middle": "data/~/backup/file.txt",
        "just_tilde": "~",
        "just_tilde_root": "/~",
        # Paths without tildes (should remain unchanged)
        "no_tilde": "/home/other/file.txt",
        "relative": "documents/file.txt",
        # Edge cases
        "tilde_not_path_separator": "~file.txt",  # Not a path separator
        "multiple_tildes_middle": "data/~/backup/~/file.txt",
    }


@pytest.fixture
def sample_urls() -> dict[str, str]:
    """Sample URL strings for tilde expansion testing.

    Returns:
        dict[str, str]: Collection of test URLs with expected expansions
    """
    return {
        # URL with tilde patterns
        "sqlite_tilde": "sqlite:///~/database.db",
        "file_tilde": "file:///~/documents/file.txt",
        "file_tilde_no_leading_slash": "file://~/documents/file.txt",
        "http_with_tilde": "http://localhost/~/api/data",
        # URLs without tildes (should remain unchanged)
        "sqlite_no_tilde": "sqlite:///var/data/database.db",
        "http_no_tilde": "http://localhost:5000/api/data",
        # Edge cases
        "scheme_only": "sqlite://",
        "empty_path": "file:///",
        "tilde_in_host": "http://~server/path",  # Should not expand
    }


@pytest.fixture
def non_string_inputs() -> list[Any]:
    """Non-string inputs that should pass through unchanged.

    Returns:
        list[Any]: Collection of non-string values
    """
    return [
        None,
        42,
        3.14,
        [],
        {},
        Path("/some/path"),
        b"bytes",
        True,
        False,
    ]


@pytest.fixture
def mock_validator_func() -> Mock:
    """Mock validator function for decorator testing.

    Returns:
        Mock: Mock validator that returns the input value
    """
    mock_func = Mock()
    mock_func.return_value = "validated_value"
    return mock_func


@pytest.fixture
def expected_path_expansions(mock_home_path: str) -> dict[str, str]:
    """Expected results for path expansions using mock home path.

    Args:
        mock_home_path: Mock home directory path

    Returns:
        dict[str, str]: Expected expansion results for paths
    """
    return {
        "~/documents/file.txt": f"{mock_home_path}/documents/file.txt",
        "~": mock_home_path,
        "/home/other/file.txt": "/home/other/file.txt",  # Unchanged
        "documents/file.txt": "documents/file.txt",  # Unchanged
        "~file.txt": "~file.txt",  # Unchanged - not a valid tilde pattern
        "/~/documents/file.txt": "/~/documents/file.txt",  # Unchanged - invalid pattern, will fail naturally
        "/~": "/~",  # Unchanged - invalid pattern, will fail naturally
    }


@pytest.fixture
def expected_url_expansions(mock_home_path: str) -> dict[str, str]:
    """Expected results for URL expansions using mock home path.

    Args:
        mock_home_path: Mock home directory path

    Returns:
        dict[str, str]: Expected expansion results for URLs
    """
    return {
        "sqlite:///~/database.db": f"sqlite:///{mock_home_path}/database.db",
        "file:///~/documents/file.txt": f"file:///{mock_home_path.lstrip('/')}/documents/file.txt",
        "file://~/documents/file.txt": f"file:///{mock_home_path.lstrip('/')}/documents/file.txt",
        "http://localhost/~/api/data": f"http://localhost/{mock_home_path}/api/data",
        "sqlite:///var/data/database.db": "sqlite:///var/data/database.db",  # Unchanged
        "http://localhost:5000/api/data": "http://localhost:5000/api/data",  # Unchanged
        "sqlite://": "sqlite://",  # Unchanged
        "file:///": "file:///",  # Unchanged
        "http://~server/path": "http://~server/path",  # Unchanged - tilde in host
    }
