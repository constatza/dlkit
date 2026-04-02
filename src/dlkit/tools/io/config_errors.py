"""Configuration error types for TOML config loading."""

from typing import Any


class ConfigSectionError(ValueError):
    """Raised when a config section is missing or invalid."""

    def __init__(
        self,
        message: str,
        section_name: str | None = None,
        available_sections: list[str] | None = None,
    ):
        super().__init__(message)
        self.section_name = section_name
        self.available_sections = available_sections or []


class ConfigValidationError(ValueError):
    """Raised when config validation fails."""

    def __init__(self, message: str, model_class: str, section_data: dict[str, Any] | None = None):
        super().__init__(message)
        self.model_class = model_class
        self.section_data = section_data or {}
