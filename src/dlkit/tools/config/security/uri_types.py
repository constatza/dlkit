"""Secure URI and path types built with Pydantic‑only validators.

This module re‑exports focused, Pydantic Annotated types for MLflow
backends, artifact destinations, tracking URLs, and secure local paths.
"""

from typing import Annotated

from pydantic import Field

from dlkit.tools.datatypes.urls import (
    ArtifactDestination,
    LocalPath,
    MLflowBackendUrl,
    MLflowTrackingUrl,
)

SecureMLflowBackendStoreUri = Annotated[
    MLflowBackendUrl,
    Field(
        description="MLflow backend store URL (sqlite:///, file://, http(s)://, s3://, gs://, wasbs://, hdfs://, databricks://)"
    ),
]

SecureMLflowTrackingUri = Annotated[
    MLflowTrackingUrl,
    Field(description="MLflow tracking URI (sqlite:///, http/https, file://, or databricks://)"),
]

SecureArtifactStoreUri = Annotated[
    ArtifactDestination,
    Field(description="Artifact destination: file://, cloud URL, or secure local path"),
]

SecurePath = Annotated[
    LocalPath,
    Field(description="Secure local path with strict tilde expansion and traversal checks"),
]
