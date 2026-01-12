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

import os
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

    if isinstance(value, os.PathLike):
        value = os.fspath(value)

    if not isinstance(value, str) or "~" not in value:
        return value

    home = _home()

    if "://" in value:
        return _expand_tilde_in_url(value, home)

    return _expand_tilde_in_path(value, home)


def local_path_security_check(value: Any) -> Any:
    """Normalize local paths while accepting os.PathLike inputs."""

    if value is None:
        return value

    if isinstance(value, os.PathLike):
        value = os.fspath(value)

    if not isinstance(value, str):
        raise TypeError("SecurePath values must be strings or path-like objects")

    return value.replace("\\", "/")


# ------------------------------
# Scheme-specific Url types
# ------------------------------

HttpUrl = Annotated[Url, UrlConstraints(allowed_schemes=["http", "https"])]

FileUrl = Annotated[Url, UrlConstraints(allowed_schemes=["file"], host_required=False)]


def _sqlite_after(val: Url) -> Url:
    """Validate SQLite URL format and path.

    Args:
        val: Pydantic URL object to validate

    Returns:
        Validated URL

    Raises:
        ValueError: If URL format is invalid or path is empty
    """
    s = str(val)

    # Guard: SQLite URL must use triple-slash form
    if not s.startswith("sqlite:///"):
        raise ValueError("SQLite URL must use 'sqlite:///' form")

    # Guard: SQLite URL must include a non-empty database path
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
    """Validate S3 bucket name against AWS naming rules.

    Args:
        name: Bucket name to validate

    Returns:
        True if valid bucket name

    Raises:
        None - returns False for invalid names
    """
    # Guard: bucket name must be between 3 and 63 characters
    if not (3 <= len(name) <= 63):
        return False

    # Guard: bucket name must be lowercase only
    if name != name.lower():
        return False

    # Guard: bucket name must match pattern (alphanumeric, dots, hyphens)
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*[a-z0-9]", name or ""):
        return False

    # Guard: bucket name must not contain invalid sequences
    if ".." in name or "-." in name or ".-" in name:
        return False

    return True


def _cloud_after(val: Url) -> Url:
    """Validate cloud storage URL (S3, GCS, WASBS, HDFS).

    Args:
        val: Pydantic URL object to validate

    Returns:
        Validated URL

    Raises:
        ValueError: If S3 bucket name is invalid
    """
    s = str(val)

    # Guard: if not S3, return unchanged (HDFS/GCS/WASBS have no additional validation)
    if not s.startswith("s3://"):
        return val

    # Extract S3 bucket name
    rest = s[len("s3://") :]
    bucket = rest.split("/", 1)[0]

    # Guard: S3 bucket name must be valid
    if not _validate_s3_bucket(bucket):
        raise ValueError("Invalid S3 bucket name")

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
            """Validate Databricks URL format.

            Args:
                value: Value to validate

            Returns:
                Validated string URL

            Raises:
                ValueError: If value is not a string or format is invalid
            """
            value = tilde_expand_strict(value)

            # Guard: Databricks URL must be a string
            if not isinstance(value, str):
                raise ValueError("Databricks URL must be a string")

            # Guard: Databricks URL must match required pattern
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
    """Validate MLflow backend URL against supported schemes.

    Args:
        value: Value to validate (string URL or other)

    Returns:
        Validated URL string or original value if not a string

    Raises:
        ValueError: If string doesn't match any supported backend URL format
    """
    # Guard: non-string values pass through unchanged
    if not isinstance(value, str):
        return value

    # Try each adapter in order
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

    # Guard: no adapter matched - raise error
    raise ValueError(
        "Invalid MLflow backend URL. Expected one of: sqlite:///, file://, http(s)://, s3://, gs://, wasbs://, hdfs://, databricks://, or common SQL backends (postgresql, mysql, mssql, oracle)"
    )


def _validate_artifact_destination(value: Any) -> Any:
    """Validate artifact destination (URL or local path).

    Args:
        value: Value to validate (string path/URL or other)

    Returns:
        Validated path/URL string or original value if not a string

    Raises:
        ValueError: If URL doesn't match supported schemes
    """
    # Guard: non-string values pass through unchanged
    if not isinstance(value, str):
        return value

    # Guard: if not a URL, treat as local path
    if "://" not in value:
        return local_path_security_check(value)

    # Try supported URL adapters
    for adapter in (_ADAPTER_FILE, _ADAPTER_CLOUD):
        out = _validate_with_adapter(adapter, value)
        if out is not None:
            return out

    # Guard: no adapter matched - raise error
    raise ValueError("Invalid artifact URL. Expected file:// or a supported cloud scheme")


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

    Args:
        value: Value to validate (string URL or other)

    Returns:
        Validated URL string or original value if not a string

    Raises:
        ValueError: If string doesn't match supported tracking URL formats
    """
    # Guard: non-string values pass through unchanged
    if not isinstance(value, str):
        return value

    # Try each supported adapter
    for adapter in (
        _ADAPTER_HTTP,
        _ADAPTER_FILE,
        _ADAPTER_DATABRICKS,
    ):
        out = _validate_with_adapter(adapter, value)
        if out is not None:
            return out

    # Guard: no adapter matched - raise error
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
