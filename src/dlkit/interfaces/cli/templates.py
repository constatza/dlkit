"""Public CLI-facing access to shared JobConfig template builders."""

from __future__ import annotations

from dlkit.infrastructure.config._template_helpers import (
    TemplateKind,
    build_inference_template_dict,
    build_mlflow_template_dict,
    build_search_template_dict,
    build_training_template_dict,
    get_template_dict,
    render_template,
    render_toml,
)

__all__ = [
    "TemplateKind",
    "build_training_template_dict",
    "build_inference_template_dict",
    "build_mlflow_template_dict",
    "build_search_template_dict",
    "get_template_dict",
    "render_toml",
    "render_template",
]
