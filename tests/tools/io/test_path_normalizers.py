"""Tests for path normalization functions in path_normalizers.py.

This module tests file path and URI normalization utilities.
SQLite URI tests have been moved to test_url_resolver.py as part of
consolidating to url_resolver as the single source of truth for URI handling.
"""

from __future__ import annotations

from pathlib import Path

from dlkit.infrastructure.io.path_normalizers import (
    canonicalize_file_path,
    normalize_file_uri,
    path_to_file_uri,
)


class TestCanonicalizeFilePath:
    """Test canonicalize_file_path function to ensure UNC path handling works correctly."""

    def test_single_leading_slash_preserved(self) -> None:
        """Test that paths with single leading slash are treated as absolute POSIX paths."""
        path = "/data/projects/test"
        result = canonicalize_file_path(path)

        assert result == "/data/projects/test"
        assert not result.startswith("//")

    def test_double_leading_slash_with_drive_letter(self) -> None:
        """Test that //C:/path patterns are correctly normalized to C:/path."""
        path = "//C:/data/test"
        result = canonicalize_file_path(path)

        # Should strip the leading slashes and normalize the drive letter
        assert result.startswith("C:")
        assert "data/test" in result

    def test_forward_slash_normalization(self) -> None:
        """Test that backslashes are converted to forward slashes."""
        path = "C:\\data\\test"
        result = canonicalize_file_path(path)

        assert "\\" not in result
        assert "/" in result


class TestNormalizeFileUri:
    """Test normalize_file_uri function."""

    def test_file_uri_normalization(self, tmp_path: Path) -> None:
        """Test that file:// URIs are normalized correctly."""
        test_path = tmp_path / "test.txt"
        uri = f"file://{test_path.as_posix()}"

        result = normalize_file_uri(uri, tmp_path)

        assert result is not None
        assert result.startswith("file://")
        assert test_path.as_posix() in result


class TestPathToFileUri:
    """Test path_to_file_uri conversion function."""

    def test_absolute_path_creates_file_uri(self, tmp_path: Path) -> None:
        """Test that absolute paths create correct file:// URIs."""
        test_path = tmp_path / "test.txt"
        result = path_to_file_uri(test_path)

        assert result.startswith("file://")
        assert tmp_path.as_posix() in result
