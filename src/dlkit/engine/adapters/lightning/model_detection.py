"""Model-family detection helpers used during assembly.

Classifies a model settings object into a :class:`ModelType` based on the
class it names. Lives in the engine layer (not domain) because
classification depends on ``lightning.pytorch``, which the domain layer must
not import.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any

import torch.nn as nn

from dlkit.infrastructure.utils.general import import_object
from dlkit.infrastructure.utils.logging_config import get_logger

logger = get_logger(__name__)


class ModelType(Enum):
    """Model type classifications."""

    SHAPE_AWARE_DLKIT = "shape_aware_dlkit"
    SHAPE_AGNOSTIC_EXTERNAL = "shape_agnostic_external"
    GRAPH = "graph"


def _get_model_class(model_settings: Any) -> type[object] | Callable[..., object] | None:
    """Resolve the model class named by ``model_settings``.

    Args:
        model_settings: Model configuration settings.

    Returns:
        Model class if available, None otherwise.
    """
    model_name = getattr(model_settings, "name", None)
    if model_name is None:
        return None

    if isinstance(model_name, type):
        return model_name

    if isinstance(model_name, str):
        try:
            return import_object(
                model_name, fallback_module=getattr(model_settings, "module_path", "")
            )
        except (ImportError, AttributeError) as exc:
            logger.debug("Could not import model class '{}': {}", model_name, exc)
            return None

    return None


def detect_model_type(model_settings: Any) -> ModelType:
    """Detect model type using class inheritance.

    Args:
        model_settings: Model configuration settings.

    Returns:
        Detected model type.
    """
    model_cls = _get_model_class(model_settings)

    if model_cls is None or not isinstance(model_cls, type):
        return ModelType.SHAPE_AGNOSTIC_EXTERNAL

    try:
        from lightning.pytorch import LightningModule

        from dlkit.domain.nn.graph.base import BaseGraphNetwork

        if issubclass(model_cls, BaseGraphNetwork):
            return ModelType.GRAPH

        if issubclass(model_cls, LightningModule):
            return ModelType.SHAPE_AGNOSTIC_EXTERNAL

        if issubclass(model_cls, nn.Module):
            return ModelType.SHAPE_AWARE_DLKIT

    except ImportError as exc:
        logger.debug("Model type detection could not import a classifier base: {}", exc)

    return ModelType.SHAPE_AGNOSTIC_EXTERNAL
