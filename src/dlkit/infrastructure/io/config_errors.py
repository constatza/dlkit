"""Configuration error types for TOML config loading.

ConfigValidationError is the canonical class; imported here for backward compatibility.
"""

from dlkit.infrastructure.config.validators import ConfigValidationError


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


__all__ = ["ConfigSectionError", "ConfigValidationError"]
