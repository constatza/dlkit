"""Pydantic‑only URL and path types for DLKit.

Design goals:
- Use Pydantic v2 primitives only (Url, UrlConstraints, Before/AfterValidator).
- Handle both remote URLs and local paths without urllib/httpx.
- Provide scheme‑specific Annotated types and composite MLflow types.
- Centralize strict tilde expansion and local path security checks.

Important:
- Do *not* import or rely on `urllib`, `requests`, `httpx`, or similar parsing helpers
  inside this module. All parsing/validation must flow through `pydantic_core.Url`
  or higher-level Pydantic utilities so behavior remains consistent across platforms.
"""

import re
from typing import Annotated, Any

from pydantic import AfterValidator, BeforeValidator, TypeAdapter
from pydantic import UrlConstraints
from pydantic_core import Url
from pydantic_core import core_schema

from dlkit.core.datatypes.tilde_expansion import (
    _expand_tilde_in_path,
    _expand_tilde_in_url,
    _home,
)


# ------------------------------
# Tilde expansion and path checks
# ------------------------------


def tilde_expand_strict(value: Any) -> Any:
    """Expand tildes using the shared tilde expansion utilities."""

    if not isinstance(value, str) or "~" not in value:
        return value

    home = _home()

    if "://" in value:
        return _expand_tilde_in_url(value, home)

    return _expand_tilde_in_path(value, home)


def local_path_security_check(value: Any) -> Any:
    """Normalize local paths via pathlib expansion."""

    if not isinstance(value, str):
        return value

    return value.replace("\\", "/")


# ------------------------------
# Scheme-specific Url types
# ------------------------------

HttpUrl = Annotated[Url, UrlConstraints(allowed_schemes=["http", "https"])]

FileUrl = Annotated[Url, UrlConstraints(allowed_schemes=["file"], host_required=False)]


def _sqlite_after(val: Url) -> Url:
    # Enforce triple-slash form and non-empty path
    s = str(val)
    if not s.startswith("sqlite:///"):
        raise ValueError("SQLite URL must use 'sqlite:///' form")
    path = s[len("sqlite:///") :]
    if path == "":
        raise ValueError("SQLite URL must include a database path")
    return val


SQLiteUrl = Annotated[
    Url,
    UrlConstraints(allowed_schemes=["sqlite"], host_required=False),
    AfterValidator(_sqlite_after),
]


def _validate_s3_bucket(name: str) -> bool:
    # Based on AWS S3 bucket naming rules (simplified, strict lowercase)
    if not (3 <= len(name) <= 63):
        return False
    if name != name.lower():
        return False
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*[a-z0-9]", name or ""):
        return False
    if ".." in name or "-." in name or ".-" in name:
        return False
    return True


def _cloud_after(val: Url) -> Url:
    s = str(val)
    # s3://bucket/prefix
    if s.startswith("s3://"):
        rest = s[len("s3://") :]
        bucket = rest.split("/", 1)[0]
        if not _validate_s3_bucket(bucket):
            raise ValueError("Invalid S3 bucket name")
    # hdfs can be hdfs:///path or hdfs://host:port/path; no extra checks here
    return val


CloudStorageUrl = Annotated[
    Url,
    UrlConstraints(allowed_schemes=["s3", "gs", "wasbs", "hdfs"], host_required=False),
    AfterValidator(_cloud_after),
]


# ------------------------------
# Common SQL database URL types
# ------------------------------

# Allow typical SQLAlchemy database backends used by MLflow server
DbUrl = Annotated[
    Url,
    UrlConstraints(
        allowed_schemes=[
            "postgresql",
            "postgresql+psycopg2",
            "mysql",
            "mysql+pymysql",
            "mssql",
            "mssql+pyodbc",
            "oracle",
            "oracle+cx_oracle",
        ],
        host_required=True,
    ),
]


# ------------------------------
# Databricks custom Pydantic type
# ------------------------------


class DatabricksUrl(str):
    """Validate databricks URLs: databricks://profile:workspace.

    Optional shorthand literal 'databricks' can be allowed by relaxing the regex
    below if desired (not enabled by default).
    """

    _re = re.compile(r"^databricks://[A-Za-z0-9._-]+:[A-Za-z0-9._/:-]+$")

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any):
        def validate(value: Any) -> str:
            value = tilde_expand_strict(value)
            if not isinstance(value, str):
                raise ValueError("Databricks URL must be a string")
            if not cls._re.match(value):
                raise ValueError("Invalid Databricks URL format")
            return value

        return core_schema.no_info_plain_validator_function(validate)


# ------------------------------
# Composite MLflow types
# ------------------------------


def _validate_with_adapter(adapter: TypeAdapter, value: str) -> str | None:
    try:
        res = adapter.validate_python(value)
        return str(res)
    except Exception:
        return None


_ADAPTER_SQLITE = TypeAdapter(SQLiteUrl)
_ADAPTER_FILE = TypeAdapter(FileUrl)
_ADAPTER_HTTP = TypeAdapter(HttpUrl)
_ADAPTER_CLOUD = TypeAdapter(CloudStorageUrl)
_ADAPTER_DATABRICKS = TypeAdapter(DatabricksUrl)
_ADAPTER_DB = TypeAdapter(DbUrl)


def _validate_mlflow_backend(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    for adapter in (
        _ADAPTER_SQLITE,
        _ADAPTER_FILE,
        _ADAPTER_HTTP,
        _ADAPTER_CLOUD,
        _ADAPTER_DATABRICKS,
        _ADAPTER_DB,
    ):
        out = _validate_with_adapter(adapter, value)
        if out is not None:
            return out
    raise ValueError(
        "Invalid MLflow backend URL. Expected one of: sqlite:///, file://, http(s)://, s3://, gs://, wasbs://, hdfs://, databricks://, or common SQL backends (postgresql, mysql, mssql, oracle)"
    )


def _validate_artifact_destination(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if "://" in value:
        for adapter in (_ADAPTER_FILE, _ADAPTER_CLOUD):
            out = _validate_with_adapter(adapter, value)
            if out is not None:
                return out
        raise ValueError("Invalid artifact URL. Expected file:// or a supported cloud scheme")
    # Local path
    value = local_path_security_check(value)
    return value


MLflowBackendUrl = Annotated[
    str,
    BeforeValidator(tilde_expand_strict),
    AfterValidator(_validate_mlflow_backend),
]


def _validate_mlflow_tracking(value: Any) -> Any:
    """Validate MLflow tracking URL.

    Accepts http(s)://, file://, and databricks:// URIs. Returns a normalized
    string representation produced by the underlying TypeAdapter when possible
    (e.g., http URLs include a trailing slash).
    """
    if not isinstance(value, str):
        return value
    for adapter in (
        _ADAPTER_HTTP,
        _ADAPTER_FILE,
        _ADAPTER_DATABRICKS,
    ):
        out = _validate_with_adapter(adapter, value)
        if out is not None:
            return out
    raise ValueError("Invalid MLflow tracking URL. Expected http(s)://, file://, or databricks://")


MLflowTrackingUrl = Annotated[
    str,
    BeforeValidator(tilde_expand_strict),
    AfterValidator(_validate_mlflow_tracking),
]

ArtifactDestination = Annotated[
    str,
    BeforeValidator(tilde_expand_strict),
    AfterValidator(_validate_artifact_destination),
]

LocalPath = Annotated[
    str,
    BeforeValidator(tilde_expand_strict),
    AfterValidator(local_path_security_check),
]


# Backward-compat exported names kept for settings
MLflowServerUri = HttpUrl
MLflowArtifactsUri = ArtifactDestination
CloudStorageUri = CloudStorageUrl
