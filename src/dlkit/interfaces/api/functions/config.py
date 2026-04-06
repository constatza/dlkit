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
from dlkit.infrastructure.config import GeneralSettings
from dlkit.infrastructure.config.protocols import BaseSettingsProtocol


def validate_config(
    settings: BaseSettingsProtocol,
    dry_build: bool = False,
) -> bool:
    """Validate configuration structure and optional runtime readiness."""
    return runtime_validate_config(settings, dry_build=dry_build)


def generate_template(
    template_type: TemplateKind = "training",
) -> str:
    """Generate configuration template."""
    _ = cast(BaseSettingsProtocol, GeneralSettings())
    return runtime_generate_template(template_type=template_type)


def validate_template(
    template_content: str,
    template_type: str | None = None,
) -> dict[str, Any]:
    """Validate configuration template."""
    template_kind: TemplateKind | None = (
        cast(TemplateKind, template_type) if template_type is not None else None
    )
    _ = cast(BaseSettingsProtocol, GeneralSettings())
    return runtime_validate_template(template_content=template_content, template_type=template_kind)
