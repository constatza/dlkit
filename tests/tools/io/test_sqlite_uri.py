"""Tests for SQLite URI handling using url_resolver.

This module contains regression tests for the SQLite URI absolute path bug
that was fixed by migrating to url_resolver as the single source of truth.

Bug History:
- Original bug (2025-11-18): path_normalizers functions were mangling SQLite URIs
  with absolute paths, causing Windows UNC path interpretation on Linux.
- Migration (2025-12-09): Migrated to url_resolver which correctly handles
  SQLite URIs with 4 slashes for absolute paths.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dlkit.infrastructure.io import url_resolver


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path


class TestSqliteUriAbsolutePaths:
    """Test that SQLite URIs with absolute paths work correctly on their native platforms."""

    @pytest.mark.skipif(os.name == "nt", reason="Unix absolute path test - runs on Unix only")
    def test_user_reported_absolute_path_bug_unix(self, root: Path) -> None:
        """Regression test for sqlite:////data/projects/... absolute path bug.

        Bug report (2025-12-09): Absolute paths with 4 slashes in config were being
        normalized to 3 slashes by the buggy path_normalizers functions, causing
        relative path interpretation instead of absolute.

        This test ensures url_resolver correctly handles Unix absolute SQLite URIs
        on Unix platforms.
        """
        # User's exact scenario: Unix absolute path from config
        uri = "sqlite:////data/projects/graph-cg/data/mlruns/mlflow.db"

        # Should preserve Unix absolute path with 4 slashes on Unix
        result = url_resolver.normalize_uri(uri, root)

        assert result.startswith("sqlite:////"), (
            f"Absolute SQLite URI must maintain 4 slashes!\n"
            f"  Input:  {uri}\n"
            f"  Output: {result}\n"
            f"  This was the bug: 4 slashes → 3 slashes incorrectly"
        )
        assert "/data/projects/graph-cg/data/mlruns/mlflow.db" in result

    @pytest.mark.skipif(os.name != "nt", reason="Windows absolute path test - runs on Windows only")
    def test_windows_absolute_path_handling(self, root: Path) -> None:
        """Test that Windows absolute paths in SQLite URIs work correctly on Windows.

        Windows absolute paths (C:/..., D:/...) should maintain 3 slashes in SQLite URIs
        since the drive letter makes them distinguishable from relative paths.
        """
        # Windows absolute path from config
        uri = "sqlite:///C:/data/projects/graph-cg/data/mlruns/mlflow.db"

        # Should preserve Windows absolute path with 3 slashes on Windows
        result = url_resolver.normalize_uri(uri, root)

        assert result.startswith("sqlite:///C:/"), (
            f"Windows absolute SQLite URI must maintain proper format!\n"
            f"  Input:  {uri}\n"
            f"  Output: {result}"
        )
        assert "C:/data/projects/graph-cg/data/mlruns/mlflow.db" in result

    @pytest.mark.skipif(os.name == "nt", reason="Unix absolute path test")
    def test_build_uri_creates_four_slashes_unix(self, root: Path) -> None:
        """Test that build_uri creates 4 slashes for Unix absolute paths."""
        absolute_path = root / "mlruns" / "mlflow.db"
        absolute_path.parent.mkdir(parents=True, exist_ok=True)

        result = url_resolver.build_uri(absolute_path, scheme="sqlite")

        assert result.startswith("sqlite:////"), (
            f"build_uri must create 4 slashes for Unix absolute paths!\n"
            f"  Path: {absolute_path}\n"
            f"  Result: {result}"
        )

    @pytest.mark.skipif(os.name != "nt", reason="Windows absolute path test")
    def test_build_uri_creates_three_slashes_windows(self, root: Path) -> None:
        """Test that build_uri creates 3 slashes for Windows absolute paths.

        Windows paths use 3 slashes because the drive letter disambiguates them.
        """
        absolute_path = root / "mlruns" / "mlflow.db"
        absolute_path.parent.mkdir(parents=True, exist_ok=True)

        result = url_resolver.build_uri(absolute_path, scheme="sqlite")

        assert result.startswith("sqlite:///") and not result.startswith("sqlite:////"), (
            f"build_uri must create 3 slashes for Windows absolute paths!\n"
            f"  Path: {absolute_path}\n"
            f"  Result: {result}"
        )

    @pytest.mark.skipif(os.name == "nt", reason="Unix absolute path test - runs on Unix only")
    def test_normalize_preserves_four_slashes_unix(self, root: Path) -> None:
        """Test that normalize_uri preserves 4 slashes for Unix absolute paths."""
        # Unix absolute URI - testing URI parsing logic, not filesystem access
        unix_path = Path("/") / "tmp" / "mlruns" / "mlflow.db"
        uri = f"sqlite:////{unix_path.as_posix().lstrip('/')}"

        result = url_resolver.normalize_uri(uri, root)

        assert result.startswith("sqlite:////"), (
            f"normalize_uri must preserve 4 slashes for Unix absolute paths!\n"
            f"  Input: {uri}\n"
            f"  Result: {result}"
        )
        assert unix_path.as_posix() in result

    def test_unix_path_on_windows_raises_error(self, root: Path) -> None:
        """Test that Unix absolute paths on Windows raise clear errors."""
        if os.name != "nt":
            return

        uri = "sqlite:////data/projects/mlruns/mlflow.db"
        with pytest.raises(ValueError, match="Unix absolute path.*cannot be used on Windows"):
            url_resolver.normalize_uri(uri, root)

    def test_windows_path_on_unix_raises_error(self, root: Path) -> None:
        """Test that Windows absolute paths on Unix raise clear errors."""
        if os.name == "nt":
            return

        uri = "sqlite:///C:/data/projects/mlruns/mlflow.db"
        with pytest.raises(ValueError, match="Windows absolute path.*cannot be used on.*platform"):
            url_resolver.normalize_uri(uri, root)


class TestSqliteUriRelativePaths:
    """Test that SQLite URIs with relative paths are normalized correctly."""

    @pytest.mark.skipif(os.name == "nt", reason="Unix relative path test")
    def test_relative_path_becomes_absolute_unix(self, root: Path) -> None:
        """Test that relative SQLite URIs become absolute with 4 slashes on Unix."""
        uri = "sqlite:///mlruns/mlflow.db"

        result = url_resolver.normalize_uri(uri, root)

        # Unix: relative paths become absolute with 4 slashes
        assert result.startswith("sqlite:////"), (
            f"Relative paths must become absolute with 4 slashes on Unix!\n"
            f"  Input: {uri}\n"
            f"  Result: {result}"
        )
        assert "mlruns/mlflow.db" in result

    @pytest.mark.skipif(os.name != "nt", reason="Windows relative path test")
    def test_relative_path_becomes_absolute_windows(self, root: Path) -> None:
        """Test that relative SQLite URIs become absolute with 3 slashes on Windows."""
        uri = "sqlite:///mlruns/mlflow.db"

        result = url_resolver.normalize_uri(uri, root)

        # Windows: relative paths become absolute with 3 slashes
        assert result.startswith("sqlite:///") and not result.startswith("sqlite:////"), (
            f"Relative paths must become absolute with 3 slashes on Windows!\n"
            f"  Input: {uri}\n"
            f"  Result: {result}"
        )
        assert "mlruns" in result and "mlflow.db" in result


class TestSqliteUriResolution:
    """Test SQLite URI path resolution."""

    @pytest.mark.skipif(os.name == "nt", reason="Unix absolute path test")
    def test_resolve_local_uri_absolute_unix(self, root: Path) -> None:
        """Test that Unix absolute SQLite URIs resolve correctly."""
        # Testing URI resolution logic with Unix absolute path
        expected_path = Path("/") / "tmp" / "mlruns" / "mlflow.db"
        uri = f"sqlite:////{expected_path.as_posix().lstrip('/')}"

        result = url_resolver.resolve_local_uri(uri, root)

        assert result == expected_path

    def test_resolve_local_uri_relative(self, root: Path) -> None:
        """Test that relative SQLite URIs resolve relative to root."""
        uri = "sqlite:///mlruns/mlflow.db"

        result = url_resolver.resolve_local_uri(uri, root)

        expected = root / "mlruns" / "mlflow.db"
        assert result == expected


class TestSqliteUriSchemeDetection:
    """Test SQLite URI scheme detection."""

    def test_scheme_detection(self) -> None:
        """Test that sqlite:// URIs are detected correctly."""
        assert url_resolver.scheme("sqlite:///mlruns/mlflow.db") == "sqlite"
        assert url_resolver.scheme("sqlite:///C:/mlflow.db") == "sqlite"

    def test_is_local_uri(self) -> None:
        """Test that sqlite:// URIs are identified as local."""
        assert url_resolver.is_local_uri("sqlite:///mlruns/mlflow.db")
        assert url_resolver.is_local_uri("sqlite:///C:/mlflow.db")
        assert not url_resolver.is_local_uri("http://example.com")
