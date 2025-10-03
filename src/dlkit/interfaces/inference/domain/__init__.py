"""Domain layer for inference - pure business logic and interfaces.

This module contains the core business logic interfaces and value objects
for the inference domain, following Domain-Driven Design principles.
"""

from .ports import (
    IModelLoader,
    IShapeInferrer,
    IInferenceExecutor,
    IModelStateManager
)
from .models import (
    ModelState,
    InferenceRequest,
    InferenceContext
)

__all__ = [
    "IModelLoader",
    "IShapeInferrer",
    "IInferenceExecutor",
    "IModelStateManager",
    "ModelState",
    "InferenceRequest",
    "InferenceContext",
]