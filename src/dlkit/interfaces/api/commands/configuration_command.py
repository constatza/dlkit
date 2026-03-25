"""Configuration command for template generation operations."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.protocols import BaseSettingsProtocol

from ..domain.errors import ConfigurationError
from ..services.configuration_service import ConfigurationService, TemplateKind
from .base import BaseCommand


class GenerateTemplateCommandInput(BaseModel):
    """Input dataflow for template generation command."""

    template_type: TemplateKind = Field(
        description="Type of template to generate (training, inference, mlflow, optuna)"
    )


class GenerateTemplateCommandOutput(BaseModel):
    """Output dataflow for template generation command."""

    template_content: str = Field(description="Generated TOML template content")
    template_type: TemplateKind = Field(description="Type of template generated")


class GenerateTemplateCommand(
    BaseCommand[GenerateTemplateCommandInput, GenerateTemplateCommandOutput, BaseSettingsProtocol]
):
    """Command for generating configuration templates.

    Follows SOLID principles:
    - Single Responsibility: Template generation only
    - Open/Closed: Extensible for new template types
    - Dependency Inversion: Uses ConfigurationService abstraction
    """

    def __init__(self, command_name: str = "generate_template") -> None:
        """Initialize template generation command."""
        super().__init__(command_name)

    def validate_input(
        self, input_data: GenerateTemplateCommandInput, settings: BaseSettingsProtocol
    ) -> None:
        """Validate template generation input.

        Args:
            input_data: Command input to validate
            settings: Configuration (not used for template generation)

        Raises:
            ConfigurationError: On validation failure
        """
        valid_types = ["training", "inference", "mlflow", "optuna"]
        if input_data.template_type not in valid_types:
            raise ConfigurationError(
                f"Invalid template type '{input_data.template_type}'. "
                f"Valid types: {', '.join(valid_types)}"
            )

    def execute(
        self,
        input_data: GenerateTemplateCommandInput,
        settings: BaseSettingsProtocol,
        **kwargs: Any,
    ) -> GenerateTemplateCommandOutput:
        """Execute template generation.

        Args:
            input_data: Template generation parameters
            settings: Configuration (not used for template generation)
            **kwargs: Additional parameters (unused)

        Returns:
            Generated template content and metadata

        Raises:
            ConfigurationError: On template generation failure
        """
        try:
            template_content = ConfigurationService.generate_template(input_data.template_type)

            return GenerateTemplateCommandOutput(
                template_content=template_content, template_type=input_data.template_type
            )

        except Exception as e:
            raise ConfigurationError(
                f"Failed to generate {input_data.template_type} template: {e}"
            ) from e


class ValidateTemplateCommandInput(BaseModel):
    """Input dataflow for template validation command."""

    template_content: str = Field(description="TOML template content to validate")
    template_type: TemplateKind | None = Field(
        default=None, description="Expected template type for validation (optional)"
    )


class ValidateTemplateCommandOutput(BaseModel):
    """Output dataflow for template validation command."""

    is_valid: bool = Field(description="Whether template is valid")
    errors: list[str] = Field(default_factory=list, description="Validation errors if any")
    template_type: TemplateKind | None = Field(
        default=None, description="Detected or specified template type"
    )


class ValidateTemplateCommand(
    BaseCommand[ValidateTemplateCommandInput, ValidateTemplateCommandOutput, BaseSettingsProtocol]
):
    """Command for validating configuration templates.

    Validates that generated templates can be parsed and loaded as Settings.
    """

    def __init__(self, command_name: str = "validate_template") -> None:
        """Initialize template validation command."""
        super().__init__(command_name)

    def validate_input(
        self, input_data: ValidateTemplateCommandInput, settings: BaseSettingsProtocol
    ) -> None:
        """Validate template validation input.

        Args:
            input_data: Command input to validate
            settings: Configuration (not used)

        Raises:
            ConfigurationError: On validation failure
        """
        if not input_data.template_content.strip():
            raise ConfigurationError("Template content cannot be empty")

    def execute(
        self,
        input_data: ValidateTemplateCommandInput,
        settings: BaseSettingsProtocol,
        **kwargs: Any,
    ) -> ValidateTemplateCommandOutput:
        """Execute template validation.

        Args:
            input_data: Template validation parameters
            settings: Configuration (not used)
            **kwargs: Additional parameters (unused)

        Returns:
            Validation result with errors if any
        """
        errors = []

        try:
            import tomlkit

            parsed = tomlkit.loads(input_data.template_content)

            try:
                GeneralSettings.model_validate(dict(parsed))
            except Exception as e:
                errors.append(f"Settings validation failed: {e}")

        except Exception as e:
            errors.append(f"TOML parsing failed: {e}")

        return ValidateTemplateCommandOutput(
            is_valid=len(errors) == 0, errors=errors, template_type=input_data.template_type
        )
