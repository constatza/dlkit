"""Typed checkpoint data transfer objects."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ModelCheckpointDTO(BaseModel):
    """Flat DTO for model settings stored in a checkpoint.

    Attributes:
        name: Model class name or type path.
        module_path: Python module path for importing the model class.
        hyper_kwargs: Non-shape constructor kwargs — the kwargs that come from
            user config. Shape kwargs (e.g. ``in_features``) are excluded here
            because they are re-derived from stored ``input_shapes``/``output_shapes``
            at inference time via ``from_context``.
        all_hyperparams: Full settings snapshot for reference / debugging.
    """

    name: str
    module_path: str
    hyper_kwargs: dict[str, Any] = Field(default_factory=dict)
    all_hyperparams: dict[str, Any] = Field(default_factory=dict)
