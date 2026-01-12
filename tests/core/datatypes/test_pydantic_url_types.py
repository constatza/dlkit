from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from dlkit.core.datatypes.urls import (
    tilde_expand_strict,
    local_path_security_check,
    SQLiteUrl,
    FileUrl,
    HttpUrl,
    CloudStorageUrl,
    DatabricksUrl,
    MLflowBackendUrl,
    ArtifactDestination,
    LocalPath,
)


class TestTildeExpansion:
    def test_plain_path_tilde(self):
        assert tilde_expand_strict("~/x").endswith("/x")
        assert tilde_expand_strict("~").endswith("")
        assert tilde_expand_strict("/~").startswith("/")

    def test_url_path_tilde_first_segment(self):
        out = tilde_expand_strict("file:///~/docs/file.txt")
        assert out.startswith("file:///") and "~" not in out

        out = tilde_expand_strict("sqlite:///~/mlflow.db")
        assert out.startswith("sqlite:///") and "~" not in out

    def test_mid_path_tilde_passes_through(self):
        """Mid-path tilde is not expanded - passes through unchanged."""
        result = tilde_expand_strict("/a/~/b")
        assert result == "/a/~/b"  # Pass through, will fail naturally when used


class TestLocalPathSecurity:
    def test_allows_traversal_strings(self):
        assert local_path_security_check("../etc/passwd") == "../etc/passwd"
        assert local_path_security_check("a/../b") == "a/../b"

    def test_allows_control_chars(self):
        assert local_path_security_check("bad\x00path") == "bad\x00path"
        assert local_path_security_check("bad\npath") == "bad\npath"

    def test_normalizes_backslashes(self):
        assert local_path_security_check(r"a\b") == "a/b"


class TestSchemeTypes:
    def test_sqlite_url(self):
        ok = TypeAdapter(SQLiteUrl).validate_python("sqlite:///db.sqlite")
        assert str(ok).startswith("sqlite:///")
        with pytest.raises(ValidationError):
            TypeAdapter(SQLiteUrl).validate_python("sqlite://db.sqlite")

    def test_file_url(self):
        ok = TypeAdapter(FileUrl).validate_python("file:///abs/path")
        assert str(ok).startswith("file:///")

    def test_http_url(self):
        ok = TypeAdapter(HttpUrl).validate_python("http://localhost:5000")
        assert str(ok).startswith("http://localhost:5000")

    def test_cloud_storage_s3(self):
        ok = TypeAdapter(CloudStorageUrl).validate_python("s3://my-bucket/prefix")
        assert str(ok).startswith("s3://my-bucket/")
        with pytest.raises(ValidationError):
            TypeAdapter(CloudStorageUrl).validate_python("s3://Invalid-Bucket..")

    def test_databricks_url(self):
        ok = TypeAdapter(DatabricksUrl).validate_python("databricks://prod:main")
        assert isinstance(ok, str) and ok.startswith("databricks://")
        with pytest.raises(ValidationError):
            TypeAdapter(DatabricksUrl).validate_python("databricks://missing")


class TestCompositeTypes:
    def test_mlflow_backend_sqlite_with_tilde(self):
        v = TypeAdapter(MLflowBackendUrl).validate_python("sqlite:///~/db.sqlite")
        assert v.startswith("sqlite:///") and "~" not in v

    def test_mlflow_backend_http(self):
        v = TypeAdapter(MLflowBackendUrl).validate_python("http://localhost:5000")
        assert v.startswith("http://localhost:5000")

    def test_mlflow_backend_s3(self):
        v = TypeAdapter(MLflowBackendUrl).validate_python("s3://bucket/prefix")
        assert v.startswith("s3://bucket/")

    def test_mlflow_backend_databricks(self):
        v = TypeAdapter(MLflowBackendUrl).validate_python("databricks://prod:main")
        assert v.startswith("databricks://")

    def test_artifact_destination_local_and_url(self, tmp_path):
        from pathlib import Path

        # Local path
        local = TypeAdapter(ArtifactDestination).validate_python("~/artifacts")
        assert Path(local).is_absolute() and "~" not in local

        # URL
        url_val = TypeAdapter(ArtifactDestination).validate_python("file:///abs/dir")
        assert url_val.startswith("file:///")

    def test_artifact_destination_windows_drive_path(self):
        sep = chr(92)
        win_path = f"C:{sep}tmp{sep}artifacts"
        val = TypeAdapter(ArtifactDestination).validate_python(win_path)
        assert val == "C:/tmp/artifacts"

    def test_artifact_destination_mid_tilde_passes_through(self):
        """Mid-path tilde is not expanded - passes through, will fail when used."""
        result = TypeAdapter(ArtifactDestination).validate_python("data/~/backup/file.txt")
        assert result == "data/~/backup/file.txt"

    def test_local_path_type(self):
        from pathlib import Path

        p = TypeAdapter(LocalPath).validate_python("~/docs")
        assert Path(p).is_absolute() and "~" not in p
