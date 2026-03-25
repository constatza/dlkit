"""Configuration validation and template functions."""

from __future__ import annotations

from typing import Any, cast

from dlkit.interfaces.api.commands import (
    GenerateTemplateCommandInput,
    ValidateTemplateCommandInput,
    ValidationCommandInput,
    get_dispatcher,
)
from dlkit.interfaces.api.services.configuration_service import TemplateKind
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.protocols import BaseSettingsProtocol

# Get the global command dispatcher
_dispatcher = get_dispatcher()


def validate_config(
    settings: BaseSettingsProtocol,
    dry_build: bool = False,
) -> bool:
    """Validate configuration structure and optional runtime readiness.

    Args:
        settings: Parsed configuration to validate.
        dry_build: When True, attempts a dry component build to catch incompatibilities.

    Returns:
        bool: True if validation succeeds.

    Raises:
        WorkflowError: On validation failure.

    Example:
        >>> from dlkit.interfaces.api import validate_config
        >>> from dlkit.tools.io import load_settings
        >>> settings = load_settings("config.toml")
        >>> validate_config(settings, dry_build=True)
        True
    """
    input_data = ValidationCommandInput(dry_build=dry_build)

    return _dispatcher.execute("validate_config", input_data, settings)


def generate_template(
    template_type: TemplateKind = "training",
) -> str:
    """Generate configuration template.

    Args:
        template_type: Type of template to generate (training, inference, mlflow, optuna).

    Returns:
        str: Generated TOML configuration template.

    Raises:
        ConfigurationError: On template generation failure.

    Example:
        >>> from dlkit.interfaces.api import generate_template
        >>> template = generate_template("training")
        >>> isinstance(template, str)
        True
    """
    input_data = GenerateTemplateCommandInput(template_type=template_type)

    # Use empty GeneralSettings since template generation doesn't need existing config
    # TODO: TYPE — GeneralSettings.SESSION: SessionSettings vs protocol's SessionSettings | None
    result = _dispatcher.execute(
        "generate_template", input_data, cast(BaseSettingsProtocol, GeneralSettings())
    )
    return result.template_content


def validate_template(
    template_content: str,
    template_type: str | None = None,
) -> dict[str, Any]:
    """Validate configuration template.

    Args:
        template_content: TOML template content to validate.
        template_type: Expected template type for validation (optional).

    Returns:
        dict: Validation result with 'is_valid' bool and 'errors' list.

    Raises:
        ConfigurationError: On validation setup failure.

    Example:
        >>> from dlkit.interfaces.api import generate_template, validate_template
        >>> template = generate_template("training")
        >>> result = validate_template(template)
        >>> result["is_valid"]
        True
    """
    template_kind: TemplateKind | None = (
        cast(TemplateKind, template_type) if template_type is not None else None
    )
    input_data = ValidateTemplateCommandInput(
        template_content=template_content, template_type=template_kind
    )

    # Use empty GeneralSettings since template validation doesn't need existing config
    # TODO: TYPE — GeneralSettings.SESSION: SessionSettings vs protocol's SessionSettings | None
    result = _dispatcher.execute(
        "validate_template", input_data, cast(BaseSettingsProtocol, GeneralSettings())
    )

    return {
        "is_valid": result.is_valid,
        "errors": result.errors,
        "template_type": result.template_type,
    }
