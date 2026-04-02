"""Tests for dlkit.tools.io.url_resolver."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dlkit.tools.io import url_resolver


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path


def test_scheme_and_is_local(root: Path) -> None:
    file_uri = url_resolver.build_uri(root / "x", scheme="file")

    assert url_resolver.scheme(file_uri) == "file"
    assert url_resolver.scheme("sqlite:///db") == "sqlite"
    assert url_resolver.scheme("plain/path") == ""
    assert url_resolver.is_local_uri(file_uri)
    assert url_resolver.is_local_uri("sqlite:///db")
    assert not url_resolver.is_local_uri("http://example.com")


def test_resolve_local_uri_matrix(root: Path, tmp_path: Path) -> None:
    # Test absolute file URI
    abs_file = tmp_path / "data"
    file_uri = url_resolver.build_uri(abs_file, scheme="file")
    assert url_resolver.resolve_local_uri(file_uri, root) == abs_file

    # Test relative file URI
    assert url_resolver.resolve_local_uri("file:data", root) == root / "data"

    # Test relative sqlite URI
    assert (
        url_resolver.resolve_local_uri("sqlite:///mlruns/db.sqlite", root)
        == root / "mlruns" / "db.sqlite"
    )

    # Test absolute sqlite URI
    abs_db = tmp_path / "mlruns" / "db.sqlite"
    sqlite_uri = url_resolver.build_uri(abs_db, scheme="sqlite")
    abs_sqlite = url_resolver.resolve_local_uri(sqlite_uri, root)
    assert abs_sqlite == abs_db

    # Test plain relative path
    assert url_resolver.resolve_local_uri("plain/path/file.txt", root) == (
        root / "plain" / "path" / "file.txt"
    )


def test_normalize_uri_matrix(root: Path, tmp_path: Path) -> None:
    # Test absolute sqlite URI
    abs_db = tmp_path / "mlruns" / "db.sqlite"
    sqlite_uri = url_resolver.build_uri(abs_db, scheme="sqlite")
    norm_sqlite_abs = url_resolver.normalize_uri(sqlite_uri, root)
    # Should preserve the absolute path
    assert str(abs_db.as_posix()) in norm_sqlite_abs

    # Test relative sqlite URI — URIs always use forward slashes regardless of platform
    norm_sqlite_rel = url_resolver.normalize_uri("sqlite:///mlruns/db.sqlite", root)
    assert "mlruns/db.sqlite" in norm_sqlite_rel

    # Test absolute file URI
    abs_file = tmp_path / "data"
    file_uri = url_resolver.build_uri(abs_file, scheme="file")
    norm_file_abs = url_resolver.normalize_uri(file_uri, root)
    assert str(abs_file.as_posix()) in norm_file_abs

    # Test relative file URI
    norm_file_rel = url_resolver.normalize_uri("file:data", root)
    assert norm_file_rel.startswith("file://")

    # Test plain path
    norm_plain = url_resolver.normalize_uri("data/db.sqlite", root)
    assert norm_plain.startswith("file://")


def test_normalize_uri_windows_drive_paths_as_file_uri(root: Path) -> None:
    windows_backslash = r"C:\Users\runneradmin\AppData\Local\Temp\mlartifacts"
    windows_forward = "C:/Users/runneradmin/AppData/Local/Temp/mlartifacts"

    norm_backslash = url_resolver.normalize_uri(windows_backslash, root)
    norm_forward = url_resolver.normalize_uri(windows_forward, root)

    assert norm_backslash == "file:///C:/Users/runneradmin/AppData/Local/Temp/mlartifacts"
    assert norm_forward == "file:///C:/Users/runneradmin/AppData/Local/Temp/mlartifacts"


@pytest.mark.skipif(os.name == "nt", reason="Off-Windows strictness only applies on non-Windows")
def test_resolve_local_uri_windows_drive_path_remains_strict_off_windows(root: Path) -> None:
    with pytest.raises(ValueError, match="Windows absolute path"):
        url_resolver.resolve_local_uri("C:/Users/runneradmin/file.txt", root)


def test_build_uri(root: Path) -> None:
    path = root / "a" / "b"

    # File URI should start with file://
    file_uri = url_resolver.build_uri(path, scheme="file")
    assert file_uri.startswith("file://")
    assert str(path.as_posix()) in file_uri

    # SQLite URI should work for absolute paths
    sqlite_uri = url_resolver.build_uri(path, scheme="sqlite")
    assert sqlite_uri.startswith("sqlite://")
    assert str(path.as_posix()) in sqlite_uri


def test_invalid_scheme_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        url_resolver.normalize_uri("s3://bucket/key", tmp_path)
    with pytest.raises(ValueError):
        url_resolver.resolve_local_uri("https://example.com", tmp_path)


class TestFileUrlValidation:
    """Test RFC 8089 compliance validation for file URLs."""

    def test_malformed_windows_path_detected(self) -> None:
        """Malformed file://C:/path should raise ConfigurationError."""
        from dlkit.shared.errors import ConfigurationError

        malformed_url = "file://C:/Users/test/file.txt"
        with pytest.raises(ConfigurationError):
            url_resolver.resolve_local_uri(malformed_url, Path.cwd())

    def test_correct_windows_path_accepted(self) -> None:
        """Correct file:///C:/path should not raise validation error."""
        from dlkit.shared.errors import ConfigurationError

        correct_url = "file:///C:/Users/test/file.txt"
        try:
            url_resolver.resolve_local_uri(correct_url, Path.cwd())
        except ConfigurationError as exc:
            raise AssertionError("Correct file URL should not raise ConfigurationError") from exc
        except ValueError:
            # Cross-platform error is expected on non-Windows (but not ConfigurationError)
            pass

    def test_file_url_with_host_rejected(self) -> None:
        """File URLs with non-empty host should raise."""
        from dlkit.shared.errors import ConfigurationError

        url_with_host = "file://hostname/path/to/file"
        with pytest.raises(ConfigurationError):
            url_resolver.resolve_local_uri(url_with_host, Path.cwd())

    @pytest.mark.skipif(os.name == "nt", reason="Unix paths not valid on Windows")
    def test_unix_file_url_accepted(self) -> None:
        """Standard Unix file:///path should work."""
        unix_path = Path("/") / "var" / "test" / "file.txt"
        unix_url = url_resolver.build_uri(unix_path, scheme="file")
        result = url_resolver.resolve_local_uri(unix_url, Path.cwd())
        assert result == unix_path.resolve()
