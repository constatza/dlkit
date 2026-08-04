"""Configuration error types for TOML config loading.

``dlkit.common.errors.ConfigValidationError`` is the canonical class; it is
re-exported here (and, separately, from ``dlkit.infrastructure.config.validators``)
for backward compatibility with existing call sites.
"""

from dlkit.common.errors import ConfigValidationError


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
