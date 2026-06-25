"""Configuration validation and template functions."""

from __future__ import annotations

from typing import Any, cast

from dlkit.engine.workflows.entrypoints import (
    TemplateKind,
)
from dlkit.engine.workflows.entrypoints import (
    generate_template as runtime_generate_template,
)
from dlkit.engine.workflows.entrypoints import (
    validate_config as runtime_validate_config,
)
from dlkit.engine.workflows.entrypoints import (
    validate_template as runtime_validate_template,
)
from dlkit.infrastructure.config.job_config import (
    InferenceJobConfig,
    JobConfig,
    SearchJobConfig,
    TrainingJobConfig,
)


def validate_config(
    settings: TrainingJobConfig | SearchJobConfig | InferenceJobConfig | JobConfig,
    dry_build: bool = False,
) -> bool:
    """Validate configuration structure and optional runtime readiness.

    Args:
        settings: Typed job config to validate.
        dry_build: If True, perform a dry build to validate runtime readiness.

    Returns:
        True if the configuration is valid.
    """
    return runtime_validate_config(settings, dry_build=dry_build)


def generate_template(
    template_type: TemplateKind = "training",
) -> str:
    """Generate configuration template.

    Args:
        template_type: Template kind (``"training"``, ``"inference"``, ``"mlflow"``, ``"optuna"``).

    Returns:
        TOML string for the requested template.
    """
    return runtime_generate_template(template_type=template_type)


def validate_template(
    template_content: str,
    template_type: str | None = None,
) -> dict[str, Any]:
    """Validate configuration template.

    Args:
        template_content: TOML string to validate.
        template_type: Optional template kind for kind-specific validation.

    Returns:
        Validation result dictionary.
    """
    template_kind: TemplateKind | None = (
        cast(TemplateKind, template_type) if template_type is not None else None
    )
    return runtime_validate_template(template_content=template_content, template_type=template_kind)
