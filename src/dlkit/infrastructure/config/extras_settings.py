"""Optional free-form EXTRAS settings.

This section is intentionally permissive for user convenience.
It is not used by core DLKit logic; consumers may read arbitrary keys.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class ExtrasSettings(BaseSettings):
    """Free-form settings container with extras allowed.

    - Optional at top level (EXTRAS may be missing)
    - Accepts any keys/values; no predefined fields
    - Parsed as a Pydantic settings model for consistency
    """

    model_config = SettingsConfigDict(
        extra="allow",
        validate_default=True,
        validate_assignment=True,
        case_sensitive=True,
        frozen=False,
    )
