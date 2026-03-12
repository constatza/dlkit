from typing import Annotated
from pydantic import BeforeValidator, Field

from .base import (
    IntHyperparameter,
    FloatHyperparameter,
    StrHyperparameter,
    Hyperparameter,
)
# Lightweight, Pydantic v2-first URL/path types (security-light)
from .tilde_expansion import expand_tilde_in_value
from .urls import (
    HttpUrl as MLflowServerUri,
    MLflowArtifactsUri,
    CloudStorageUri,
)

# Simple Pydantic v2 Annotated shortcuts (ignore security risks by design)
# - Expand '~' early, defer all other validation to downstream consumers.
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
    "IntHyperparameter",
    "FloatHyperparameter",
    "StrHyperparameter",
    "Hyperparameter",
    # Simple shortcuts (security-light)
    "SimpleTildePath",
    "SimpleMLflowURI",
    # Back-compat specialized URL types
    "MLflowServerUri",
    "MLflowArtifactsUri",
    "CloudStorageUri",
]
