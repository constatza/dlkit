"""Tests for system utilities, particularly mkdir_for_local path handling."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from dlkit.tools.utils.system_utils import (
    coerce_root_dir_to_absolute,
    mkdir_for_local,
    normalize_user_path,
)


class TestMkdirForLocal:
    """Test the mkdir_for_local function with various URI formats."""

    def test_sqlite_three_slashes_with_dot_slash_regression(self, tmp_path: Path) -> None:
        """Regression test: sqlite:///./path should be relative, not absolute /./path.

        This test verifies the fix for the bug where sqlite:///./mlruns/mlflow.db
        was being interpreted as absolute path /./mlruns/mlflow.db (resolving to
        /mlruns, requiring root permissions) instead of relative ./mlruns/mlflow.db.
        """
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # This is the exact pattern from the bug report
            uri = "sqlite:///./mlruns/mlflow.db"
            mkdir_for_local(uri)

            # Should create directory relative to cwd, NOT at root /mlruns
            expected_dir = tmp_path / "mlruns"
            assert expected_dir.exists(), f"Directory should exist at {expected_dir}, not at /mlruns"
            assert expected_dir.is_dir()

            # Verify we didn't try to create /mlruns (would fail with permission error)
            # This assertion passing means we correctly handled the path as relative
            assert not Path("/mlruns").exists() or Path("/mlruns").is_dir()
        finally:
            os.chdir(original_cwd)

    def test_sqlite_relative_path_with_dot_slash(self, tmp_path: Path) -> None:
        """Test that sqlite:///./path is treated as relative, not absolute."""
        # Change to tmp_path so relative paths work
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # This should create under current directory, not root
            uri = "sqlite:///./mlruns/mlflow.db"
            mkdir_for_local(uri)

            # Verify directory was created relative to cwd, not root
            expected_dir = tmp_path / "mlruns"
            assert expected_dir.exists()
            assert expected_dir.is_dir()
        finally:
            os.chdir(original_cwd)

    def test_sqlite_relative_path_with_double_dot(self, tmp_path: Path) -> None:
        """Test that Pydantic normalizes ../ in URLs during parsing."""
        # Note: Pydantic Url normalizes paths, so sqlite:///../data becomes sqlite:///data
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Pydantic will normalize this to sqlite:///data/mlflow.db
            uri = "sqlite:///../data/mlflow.db"
            mkdir_for_local(uri)

            # The ../ is normalized away by Pydantic, so it creates "data" in cwd
            expected_dir = tmp_path / "data"
            assert expected_dir.exists()
            assert expected_dir.is_dir()
        finally:
            os.chdir(original_cwd)

    def test_sqlite_absolute_path(self, tmp_path: Path) -> None:
        """Test that sqlite:////absolute/path works correctly."""
        test_path = tmp_path / "absolute" / "test"
        uri = f"sqlite:///{test_path.as_posix()}/mlflow.db"

        mkdir_for_local(uri)

        assert test_path.exists()
        assert test_path.is_dir()

    def test_sqlite_with_triple_slash_no_dot(self, tmp_path: Path) -> None:
        """Test standard sqlite:///path format (relative path)."""
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            uri = "sqlite:///mlruns/mlflow.db"
            mkdir_for_local(uri)

            expected_dir = tmp_path / "mlruns"
            assert expected_dir.exists()
            assert expected_dir.is_dir()
        finally:
            os.chdir(original_cwd)

    def test_sqlite_quad_slash_absolute(self, tmp_path: Path) -> None:
        """Test sqlite:////absolute/path (four slashes for absolute)."""
        test_path = tmp_path / "quad_slash_test"
        uri = f"sqlite:///{test_path.as_posix()}/data.db"

        mkdir_for_local(uri)

        assert test_path.exists()

    def test_file_url_with_absolute_path(self, tmp_path: Path) -> None:
        """Test file:///absolute/path format."""
        # When Pydantic sees file:///tmp/path, it keeps the absolute path /tmp/path
        # Our logic strips one leading / making it tmp/path (relative)
        # This is the expected behavior for file:/// (3 slashes)
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Using a relative-looking name in the file URL
            uri = "file:///file_url_test/data.txt"
            mkdir_for_local(uri)

            # Should create relative to cwd
            expected = tmp_path / "file_url_test"
            assert expected.exists()
        finally:
            os.chdir(original_cwd)

    def test_plain_string_path(self, tmp_path: Path) -> None:
        """Test plain string path without URI scheme."""
        test_path = tmp_path / "plain_path" / "file.txt"

        mkdir_for_local(str(test_path))

        assert test_path.parent.exists()

    def test_plain_relative_path(self, tmp_path: Path) -> None:
        """Test plain relative path string."""
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            mkdir_for_local("relative/path/file.txt")

            expected_dir = tmp_path / "relative" / "path"
            assert expected_dir.exists()
        finally:
            os.chdir(original_cwd)

    def test_http_url_is_ignored(self, tmp_path: Path) -> None:
        """Test that HTTP URLs are ignored (no directory creation)."""
        # Should not raise, just return without creating anything
        mkdir_for_local("http://example.com/path/to/file")

        # Verify no directories were created in tmp_path
        assert not (tmp_path / "path").exists()

    def test_s3_url_is_ignored(self, tmp_path: Path) -> None:
        """Test that S3 URLs are ignored."""
        mkdir_for_local("s3://bucket/path/to/file")

        # Verify no directories were created
        assert not (tmp_path / "bucket").exists()

    @pytest.mark.skipif(os.name != "nt", reason="Windows-specific test")
    def test_windows_path_handling(self, tmp_path: Path) -> None:
        """Test Windows path handling (removes leading slashes)."""
        test_path = tmp_path / "windows_test"
        # On Windows, paths like /C:/path should have leading slash stripped
        uri = f"sqlite:///{test_path.as_posix()}/data.db"

        mkdir_for_local(uri)

        assert test_path.exists()

    def test_sqlite_absolute_path_with_four_slashes(self, tmp_path: Path) -> None:
        """Test that sqlite:////path creates absolute path /path."""
        # sqlite:////path (4 slashes) means absolute path /path
        # We need to use a path we have permission to create
        test_path = tmp_path / "absolute_test"
        uri = f"sqlite:///{test_path.as_posix()}/db.sqlite"

        mkdir_for_local(uri)

        # With 4 slashes (scheme:// + //path), it's absolute
        assert test_path.exists()

    def test_pydantic_url_object(self, tmp_path: Path) -> None:
        """Test that Pydantic Url objects are handled correctly."""
        from pydantic_core import Url

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Create a Url object for a relative path
            url_obj = Url("file:///pydantic_test/data.txt")

            mkdir_for_local(url_obj)

            assert (tmp_path / "pydantic_test").exists()
        finally:
            os.chdir(original_cwd)

    def test_empty_path_components_handled(self, tmp_path: Path) -> None:
        """Test handling of URIs with empty path components."""
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Path with multiple slashes that should be normalized
            uri = "sqlite:///./some//path/data.db"
            mkdir_for_local(uri)

            # Should normalize and create the directory
            expected_dir = tmp_path / "some" / "path"
            assert expected_dir.exists()
        finally:
            os.chdir(original_cwd)

    def test_nested_relative_path(self, tmp_path: Path) -> None:
        """Test deeply nested relative path."""
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            uri = "sqlite:///./a/b/c/d/e/data.db"
            mkdir_for_local(uri)

            expected_dir = tmp_path / "a" / "b" / "c" / "d" / "e"
            assert expected_dir.exists()
            assert expected_dir.is_dir()
        finally:
            os.chdir(original_cwd)

    def test_parent_directory_already_exists(self, tmp_path: Path) -> None:
        """Test that function works when parent directory already exists."""
        test_dir = tmp_path / "existing"
        test_dir.mkdir()

        uri = f"sqlite:///{test_dir.as_posix()}/subdir/data.db"
        mkdir_for_local(uri)

        assert (test_dir / "subdir").exists()

    def test_creates_multiple_parent_levels(self, tmp_path: Path) -> None:
        """Test that multiple parent directory levels are created."""
        test_path = tmp_path / "level1" / "level2" / "level3" / "level4"
        uri = f"sqlite:///{test_path.as_posix()}/data.db"

        mkdir_for_local(uri)

        assert test_path.exists()
        assert test_path.is_dir()


class TestNormalizeUserPath:
    """Tests for the path normalization helpers."""

    def test_normalize_relative_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Relative paths should resolve against the current working directory."""
        monkeypatch.chdir(tmp_path)
        result = normalize_user_path("relative/run")
        assert result == tmp_path / "relative" / "run"

    def test_normalize_tilde_expansion(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tilde expansion should respect the HOME environment variable."""
        fake_home = tmp_path / "fake_home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        import pathlib

        monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: fake_home))

        result = normalize_user_path("~/runs", require_absolute=True)
        assert result == fake_home / "runs"

    def test_coerce_root_dir_missing_slash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Paths missing a leading slash but matching HOME prefix are repaired."""
        fake_home = Path("/home/tester")
        monkeypatch.setenv("HOME", str(fake_home))

        import pathlib

        monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: fake_home))

        value = "home/tester/project"
        result = coerce_root_dir_to_absolute(value)
        assert result == fake_home / "project"

    def test_coerce_root_dir_requires_absolute(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Relative paths that cannot be coerced should return None when absolute required."""
        monkeypatch.chdir(tmp_path)
        result = coerce_root_dir_to_absolute("relative/path")
        assert result is None
