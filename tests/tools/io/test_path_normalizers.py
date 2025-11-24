"""Tests for path normalization functions in path_normalizers.py.

This module focuses on testing URI normalization, particularly the fix
for the SQLite URI path mangling bug where absolute paths were incorrectly
interpreted as Windows UNC network paths on Linux systems.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlkit.tools.io.path_normalizers import (
    canonicalize_file_path,
    normalize_file_uri,
    normalize_sqlite_uri,
    path_to_file_uri,
    path_to_sqlite_uri,
)


# =============================================================================
# PERMANENT REGRESSION TEST - DO NOT REMOVE
# =============================================================================
# Bug Report Date: 2025-11-18
# Issue: SQLite URI Path Mangling on Linux
# Symptom: Absolute paths in SQLite URIs were corrupted with Windows UNC paths
# Example: sqlite:////data/projects/... → sqlite:///\\\\DATA\\PROJECTS/...
# Root Cause: Pydantic parses sqlite:////path as url.path = "//path", which
#             PureWindowsPath interpreted as UNC network path \\\\server\\share
# Fix: Strip one leading slash before canonicalization to prevent UNC interpretation
# Location: src/dlkit/tools/io/path_normalizers.py:normalize_sqlite_uri()
# =============================================================================


@pytest.mark.regression
class TestSqliteUriPathManglingRegression:
    """PERMANENT REGRESSION TESTS - Critical bug fixed 2025-11-18.

    DO NOT REMOVE OR MODIFY THESE TESTS.

    These tests prevent regression of a critical bug where SQLite URIs with
    absolute paths were being corrupted on Linux systems due to Windows UNC
    path interpretation.

    Bug Details:
    - Input:  sqlite:////data/projects/graph-cg/data/mlruns/mlflow.db
    - Output: sqlite:///\\\\DATA\\PROJECTS/graph-cg/data/mlruns/mlflow.db (WRONG)
    - Cause:  Double-slash prefix interpreted as Windows UNC path
    - Impact: MLflow server failed to start with mangled backend_store_uri
    """

    def test_exact_bug_report_scenario(self, tmp_path: Path) -> None:
        """REGRESSION: Test exact path from bug report (2025-11-18).

        This test MUST NEVER be removed. It ensures the specific path from
        the bug report is correctly handled.

        Original Bug:
        - Input:  sqlite:////data/projects/graph-cg/data/mlruns/mlflow.db
        - Output: sqlite:///\\\\DATA\\PROJECTS/graph-cg/data/mlruns/mlflow.db
        - Issue:  Windows UNC path interpretation on Linux

        Expected Behavior:
        - Input:  sqlite:////data/projects/graph-cg/data/mlruns/mlflow.db
        - Output: sqlite:///data/projects/graph-cg/data/mlruns/mlflow.db
        """
        # Create the exact directory structure from bug report
        bug_path = tmp_path / "data" / "projects" / "graph-cg" / "data" / "mlruns"
        bug_path.mkdir(parents=True, exist_ok=True)

        # Exact URI from bug report (using tmp_path as base for testing)
        test_db = bug_path / "mlflow.db"
        uri = f"sqlite:///{test_db.as_posix()}"

        # Normalize the URI
        result = normalize_sqlite_uri(uri, tmp_path)

        # CRITICAL ASSERTIONS - These must ALWAYS pass
        assert result is not None, "normalize_sqlite_uri returned None"

        # 1. NO Windows-style backslashes (this was the primary symptom)
        assert "\\\\" not in result, (
            f"REGRESSION DETECTED: Found Windows UNC backslashes in result.\n"
            f"  Input:  {uri}\n"
            f"  Output: {result}\n"
            f"  This indicates the bug has returned!"
        )

        # 2. NO uppercase directory mangling (DATA, PROJECTS)
        assert "DATA" not in result, (
            f"REGRESSION DETECTED: Found uppercase 'DATA' in result.\n"
            f"  Input:  {uri}\n"
            f"  Output: {result}\n"
            f"  Directory names should preserve original case."
        )

        assert "PROJECTS" not in result, (
            f"REGRESSION DETECTED: Found uppercase 'PROJECTS' in result.\n"
            f"  Input:  {uri}\n"
            f"  Output: {result}\n"
            f"  Directory names should preserve original case."
        )

        # 3. Path structure must be preserved correctly
        assert "/data/projects/graph-cg/data/mlruns/mlflow.db" in result, (
            f"REGRESSION DETECTED: Path structure not preserved.\n"
            f"  Input:  {uri}\n"
            f"  Output: {result}\n"
            f"  Expected to contain: /data/projects/graph-cg/data/mlruns/mlflow.db"
        )

        # 4. Only forward slashes (no backslashes at all)
        assert result.count("\\") == 0, (
            f"REGRESSION DETECTED: Found backslash characters in result.\n"
            f"  Input:  {uri}\n"
            f"  Output: {result}\n"
            f"  All path separators should be forward slashes on Linux."
        )

    def test_various_absolute_paths_not_mangled(self, tmp_path: Path) -> None:
        """REGRESSION: Ensure various absolute path patterns work correctly.

        This test covers common absolute path patterns that could trigger
        the UNC path bug.
        """
        test_cases = [
            "/data/mlruns/mlflow.db",
            "/projects/myproject/mlflow.db",
            "/home/user/experiments/mlflow.db",
            "/var/lib/mlflow/mlflow.db",
            "/opt/mlflow/backend.db",
        ]

        for path_str in test_cases:
            # Create directory
            path_obj = tmp_path / path_str.lstrip("/")
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            # Create URI (sqlite:////absolute/path)
            uri = f"sqlite:///{path_obj.as_posix()}"

            # Normalize
            result = normalize_sqlite_uri(uri, tmp_path)

            # Verify no UNC corruption
            assert result is not None
            assert "\\\\" not in result, f"UNC backslashes found for path: {path_str}"
            assert result.count("\\") == 0, f"Backslashes found for path: {path_str}"

            # Verify path is in result (may be relative to tmp_path)
            path_components = Path(path_str).parts
            # Check that at least some of the path components are in the result
            assert any(part.lower() in result.lower() for part in path_components if part != "/"), (
                f"Path components not found in result for: {path_str}\n"
                f"  Result: {result}"
            )

    def test_unc_interpretation_prevented(self, tmp_path: Path) -> None:
        """REGRESSION: Verify that double-slash paths don't trigger UNC logic.

        This test specifically checks that the fix (stripping one leading slash)
        prevents the UNC path interpretation that caused the bug.
        """
        # These paths start with // after URI parsing, which triggered the bug
        test_path = tmp_path / "data" / "test.db"
        test_path.parent.mkdir(parents=True, exist_ok=True)

        uri = f"sqlite:///{test_path.as_posix()}"
        result = normalize_sqlite_uri(uri, tmp_path)

        assert result is not None

        # The bug manifested as treating //data/... as UNC \\\\DATA\\...
        # Verify this doesn't happen
        path_part = result.replace("sqlite:///", "").replace("sqlite://", "")

        # Should not have UNC-style backslashes at the start
        assert not path_part.startswith("\\\\"), (
            f"UNC path interpretation detected!\n"
            f"  URI: {uri}\n"
            f"  Result: {result}\n"
            f"  Path part starts with UNC backslashes"
        )

        # Should not have uppercase drive portion (UNC interpretation artifact)
        first_dir = path_part.lstrip("/").split("/")[0]
        assert not first_dir.isupper() or len(first_dir) <= 2, (
            f"Directory appears to be uppercased (UNC artifact)!\n"
            f"  URI: {uri}\n"
            f"  Result: {result}\n"
            f"  First directory: {first_dir}"
        )


class TestNormalizeSqliteUri:
    """Test normalize_sqlite_uri function, particularly the UNC path bug fix."""

    def test_absolute_path_not_mangled_on_linux(self, tmp_path: Path) -> None:
        """Regression test: sqlite:////absolute/path should not be treated as UNC path.

        This is the primary regression test for the bug where paths like
        sqlite:////data/projects/graph-cg/data/mlruns/mlflow.db were being
        corrupted to sqlite:///\\\\DATA\\PROJECTS/graph-cg/data/mlruns/mlflow.db
        on Linux systems.

        The bug occurred because Pydantic parses sqlite:////absolute/path as
        url.path = "//absolute/path" (2 leading slashes), which was being
        misinterpreted as a Windows UNC network path (\\\\server\\share).
        """
        # Use a real absolute path that exists
        test_dir = tmp_path / "data" / "projects" / "graph-cg" / "data" / "mlruns"
        test_dir.mkdir(parents=True)
        test_db = test_dir / "mlflow.db"

        # SQLite absolute path URI uses 4 slashes: sqlite:////absolute/path
        uri = f"sqlite:///{test_db.as_posix()}"

        result = normalize_sqlite_uri(uri, tmp_path)

        # Result should NOT have:
        # - Windows-style backslashes (\\)
        # - Uppercased directory names (DATA/PROJECTS)
        # - Mixed forward/backward slashes
        assert result is not None
        assert "\\\\" not in result, f"Result contains backslashes: {result}"
        assert "DATA" not in result, f"Result contains uppercase DATA: {result}"
        assert "PROJECTS" not in result, f"Result contains uppercase PROJECTS: {result}"

        # Result should maintain the correct path structure
        assert "/data/projects/graph-cg/data/mlruns/mlflow.db" in result
        # All slashes should be forward slashes
        assert result.count("\\") == 0

    def test_specific_bug_report_path(self, tmp_path: Path) -> None:
        """Test the exact path from the bug report.

        Bug report path: sqlite:////data/projects/graph-cg/data/mlruns/mlflow.db
        Expected: Correct handling without UNC interpretation
        Actual (before fix): sqlite:///\\\\DATA\\PROJECTS/graph-cg/data/mlruns/mlflow.db
        """
        # Create the directory structure from the bug report
        test_dir = tmp_path / "data" / "projects" / "graph-cg" / "data" / "mlruns"
        test_dir.mkdir(parents=True)

        # Construct the exact URI from the bug report
        # Note: We use tmp_path as the base, but the logic should still work
        absolute_path = test_dir / "mlflow.db"
        uri = f"sqlite:///{absolute_path.as_posix()}"

        result = normalize_sqlite_uri(uri, tmp_path)

        assert result is not None
        # The path should preserve correct capitalization
        assert "/data/" in result.lower()
        assert "/projects/" in result.lower()
        # Should not have Windows UNC backslashes
        assert not result.startswith("sqlite:///\\\\")
        # Should not have uppercased directories
        path_part = result.replace("sqlite:///", "").replace("sqlite://", "")
        # Check that the directories maintain lowercase (or at least not all uppercase)
        assert "DATA" not in path_part or "data" in path_part

    def test_relative_path_unaffected(self, tmp_path: Path) -> None:
        """Test that relative SQLite URIs are not affected by the fix.

        Relative paths use sqlite:///relative/path (3 slashes), which parse
        as url.path = "/relative/path" (1 leading slash), so they should not
        trigger the UNC path logic.
        """
        uri = "sqlite:///mlruns/mlflow.db"
        result = normalize_sqlite_uri(uri, tmp_path)

        assert result is not None
        # Should resolve relative to tmp_path
        assert "mlruns/mlflow.db" in result

    def test_windows_drive_letter_paths(self, tmp_path: Path) -> None:
        """Test that Windows paths with drive letters work correctly.

        This ensures the fix doesn't break Windows path handling.
        """
        # Test a Windows-style path (works on both platforms in our tests)
        uri = "sqlite:///C:/mlruns/mlflow.db"
        result = normalize_sqlite_uri(uri, tmp_path)

        assert result is not None
        # Should contain the drive letter handling
        assert "C:" in result or "c:" in result

    def test_preserve_path_separators(self, tmp_path: Path) -> None:
        """Test that all path separators remain forward slashes after normalization."""
        test_dir = tmp_path / "test" / "nested" / "path"
        test_dir.mkdir(parents=True)

        uri = f"sqlite:///{test_dir.as_posix()}/db.sqlite"
        result = normalize_sqlite_uri(uri, tmp_path)

        assert result is not None
        # All separators should be forward slashes
        path_part = result.replace("sqlite://", "")
        assert "\\" not in path_part, f"Found backslash in: {result}"

    def test_no_uppercase_mangling(self, tmp_path: Path) -> None:
        """Test that lowercase directory names remain lowercase.

        The bug caused directories to be uppercased (e.g., /data → DATA).
        This test ensures that doesn't happen.
        """
        test_dir = tmp_path / "lowercase" / "directory" / "names"
        test_dir.mkdir(parents=True)

        uri = f"sqlite:///{test_dir.as_posix()}/db.sqlite"
        result = normalize_sqlite_uri(uri, tmp_path)

        assert result is not None
        # Lowercase directory names should be preserved
        assert "lowercase" in result
        assert "directory" in result
        assert "names" in result
        # Should not have uppercase versions
        assert "LOWERCASE" not in result
        assert "DIRECTORY" not in result


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


class TestPathToSqliteUri:
    """Test path_to_sqlite_uri conversion function."""

    def test_absolute_path_creates_correct_uri(self, tmp_path: Path) -> None:
        """Test that absolute paths create sqlite:////path URIs (4 slashes)."""
        test_path = tmp_path / "test.db"
        result = path_to_sqlite_uri(test_path)

        assert result.startswith("sqlite://")
        # For absolute paths, the URI should be sqlite:///absolute/path
        # which means the path component starts with /
        assert tmp_path.as_posix() in result

    def test_relative_path_resolves_first(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that relative paths are resolved before URI conversion."""
        monkeypatch.chdir(tmp_path)
        relative_path = Path("relative/test.db")

        result = path_to_sqlite_uri(relative_path)

        assert result.startswith("sqlite://")
        # Should contain the resolved absolute path
        assert tmp_path.as_posix() in result


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
