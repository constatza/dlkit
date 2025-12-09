"""Tests for dlkit.tools.io.url_resolver."""

from __future__ import annotations

from pathlib import Path

import pytest

from dlkit.tools.io import url_resolver


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    return tmp_path


def test_scheme_and_is_local() -> None:
    assert url_resolver.scheme("file:///tmp/x") == "file"
    assert url_resolver.scheme("sqlite:///db") == "sqlite"
    assert url_resolver.scheme("/tmp/x") == ""
    assert url_resolver.is_local_uri("file:///tmp/x")
    assert url_resolver.is_local_uri("sqlite:///db")
    assert not url_resolver.is_local_uri("http://example.com")


def test_resolve_local_uri_matrix(root: Path) -> None:
    assert url_resolver.resolve_local_uri("file:///tmp/data", root) == Path("/tmp/data")
    assert url_resolver.resolve_local_uri("file:data", root) == root / "data"
    assert url_resolver.resolve_local_uri("sqlite:///mlruns/db.sqlite", root) == root / "mlruns" / "db.sqlite"
    abs_sqlite = url_resolver.resolve_local_uri("sqlite:////tmp/mlruns/db.sqlite", root)
    assert abs_sqlite == Path("/tmp/mlruns/db.sqlite")
    assert url_resolver.resolve_local_uri("plain/path/file.txt", root) == (root / "plain" / "path" / "file.txt")


def test_normalize_uri_matrix(root: Path) -> None:
    norm_sqlite_abs = url_resolver.normalize_uri("sqlite:////tmp/mlruns/db.sqlite", root)
    assert norm_sqlite_abs.startswith("sqlite:////tmp/mlruns/db.sqlite")
    norm_sqlite_rel = url_resolver.normalize_uri("sqlite:///mlruns/db.sqlite", root)
    assert norm_sqlite_rel.startswith("sqlite:////")
    assert "/mlruns/db.sqlite" in norm_sqlite_rel
    norm_file_abs = url_resolver.normalize_uri("file:///tmp/data", root)
    assert norm_file_abs == "file:///tmp/data"
    norm_file_rel = url_resolver.normalize_uri("file:data", root)
    assert norm_file_rel.startswith("file:///")
    norm_plain = url_resolver.normalize_uri("data/db.sqlite", root)
    assert norm_plain.startswith("file:///")


def test_build_uri(root: Path) -> None:
    path = root / "a" / "b"
    assert url_resolver.build_uri(path, scheme="file").startswith("file:///")
    sqlite_uri = url_resolver.build_uri(path, scheme="sqlite")
    assert sqlite_uri.startswith("sqlite:////")


def test_invalid_scheme_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        url_resolver.normalize_uri("s3://bucket/key", tmp_path)
    with pytest.raises(ValueError):
        url_resolver.resolve_local_uri("https://example.com", tmp_path)
