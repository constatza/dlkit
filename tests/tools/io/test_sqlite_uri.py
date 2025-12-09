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

from pathlib import Path

import pytest

from dlkit.tools.io import url_resolver


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    return tmp_path


class TestSqliteUriAbsolutePaths:
    """Test that SQLite URIs with absolute paths use 4 slashes."""

    def test_user_reported_absolute_path_bug(self, root: Path) -> None:
        """Regression test for sqlite:////data/projects/... absolute path bug.

        Bug report (2025-12-09): Absolute paths with 4 slashes in config were being
        normalized to 3 slashes by the buggy path_normalizers functions, causing
        relative path interpretation instead of absolute.

        This test ensures url_resolver correctly handles absolute SQLite URIs.
        """
        # User's exact scenario: absolute path from config
        uri = "sqlite:////data/projects/graph-cg/data/mlruns/mlflow.db"

        # Should preserve absolute path with 4 slashes
        result = url_resolver.normalize_uri(uri, root)

        assert result.startswith("sqlite:////"), (
            f"Absolute SQLite URI must maintain 4 slashes!\n"
            f"  Input:  {uri}\n"
            f"  Output: {result}\n"
            f"  This was the bug: 4 slashes → 3 slashes incorrectly"
        )
        assert "/data/projects/graph-cg/data/mlruns/mlflow.db" in result

    def test_build_uri_creates_four_slashes(self, root: Path) -> None:
        """Test that build_uri creates 4 slashes for absolute paths."""
        absolute_path = root / "mlruns" / "mlflow.db"
        absolute_path.parent.mkdir(parents=True, exist_ok=True)

        result = url_resolver.build_uri(absolute_path, scheme="sqlite")

        assert result.startswith("sqlite:////"), (
            f"build_uri must create 4 slashes for absolute paths!\n"
            f"  Path: {absolute_path}\n"
            f"  Result: {result}"
        )

    def test_normalize_preserves_four_slashes(self, root: Path) -> None:
        """Test that normalize_uri preserves 4 slashes for absolute paths."""
        # Create absolute URI
        uri = "sqlite:////tmp/mlruns/mlflow.db"

        result = url_resolver.normalize_uri(uri, root)

        assert result.startswith("sqlite:////"), (
            f"normalize_uri must preserve 4 slashes for absolute paths!\n"
            f"  Input: {uri}\n"
            f"  Result: {result}"
        )


class TestSqliteUriRelativePaths:
    """Test that SQLite URIs with relative paths use 3 slashes."""

    def test_relative_path_has_three_slashes(self, root: Path) -> None:
        """Test that relative SQLite URIs have 3 slashes."""
        uri = "sqlite:///mlruns/mlflow.db"

        result = url_resolver.normalize_uri(uri, root)

        # After normalization relative to root, should become absolute with 4 slashes
        assert result.startswith("sqlite:////"), (
            f"Relative paths normalized against root become absolute with 4 slashes!\n"
            f"  Input: {uri}\n"
            f"  Result: {result}"
        )
        assert "mlruns/mlflow.db" in result


class TestSqliteUriResolution:
    """Test SQLite URI path resolution."""

    def test_resolve_local_uri_absolute(self, root: Path) -> None:
        """Test that absolute SQLite URIs resolve correctly."""
        uri = "sqlite:////tmp/mlruns/mlflow.db"

        result = url_resolver.resolve_local_uri(uri, root)

        assert result == Path("/tmp/mlruns/mlflow.db")

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
        assert url_resolver.scheme("sqlite:////tmp/mlflow.db") == "sqlite"

    def test_is_local_uri(self) -> None:
        """Test that sqlite:// URIs are identified as local."""
        assert url_resolver.is_local_uri("sqlite:///mlruns/mlflow.db")
        assert url_resolver.is_local_uri("sqlite:////tmp/mlflow.db")
        assert not url_resolver.is_local_uri("http://example.com")
