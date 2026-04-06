"""Infrastructure value types: URL/path types and data-split utilities."""

from typing import Annotated

from pydantic import BeforeValidator, Field

from .split import IndexSplit, Splitter
from .tilde_expansion import expand_tilde_in_value
from .urls import (
    ArtifactDestination,
    CloudStorageUri,
    CloudStorageUrl,
    DatabricksUrl,
    DbUrl,
    FileUrl,
    HttpUrl,
    LocalPath,
    MLflowArtifactsUri,
    MLflowBackendUrl,
    MLflowServerUri,
    MLflowTrackingUrl,
    SQLiteUrl,
    local_path_security_check,
    tilde_expand_strict,
)

SimpleTildePath = Annotated[
    str,
    BeforeValidator(expand_tilde_in_value),
    Field(description="Path string with pre-validation ~ expansion (no extra checks)"),
]

SimpleMLflowURI = Annotated[
    str,
    BeforeValidator(expand_tilde_in_value),
    Field(description="Generic MLflow-style URI with pre-validation ~ expansion only"),
]

__all__ = [
    "ArtifactDestination",
    "CloudStorageUri",
    "CloudStorageUrl",
    "DatabricksUrl",
    "DbUrl",
    "FileUrl",
    "HttpUrl",
    "IndexSplit",
    "LocalPath",
    "MLflowArtifactsUri",
    "MLflowBackendUrl",
    "MLflowServerUri",
    "MLflowTrackingUrl",
    "SimpleMLflowURI",
    "SimpleTildePath",
    "SQLiteUrl",
    "Splitter",
    "expand_tilde_in_value",
    "local_path_security_check",
    "tilde_expand_strict",
]
