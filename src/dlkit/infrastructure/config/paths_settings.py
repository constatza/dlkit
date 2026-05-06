"""Optional paths settings with automatic path resolution.

This section provides standardized path configuration with DLKit's automatic
path resolution. All paths are resolved relative to the config's root_dir
following DLKit's standard path resolution hierarchy.

Common use cases:
- Data file paths (matrix, vectors, datasets)
- Model checkpoints and weights
- Output directories
- Custom user-defined paths

All fields are optional and extras are allowed for maximum flexibility.
"""

from __future__ import annotations

from typing import Self

from pydantic import Field, TypeAdapter, ValidationError, model_validator
from pydantic_settings import SettingsConfigDict

from dlkit.infrastructure.config.security.uri_types import SecurePath

from .core.base_settings import BasicSettings

_PATH_VALUE_ADAPTER = TypeAdapter(SecurePath | None)


class PathsSettings(BasicSettings):
    """Optional paths configuration with automatic resolution and extras support.

    Provides standardized path fields with DLKit's automatic path resolution.
    All relative paths are resolved against SESSION.root_dir following DLKit's
    standard path resolution hierarchy.

    Features:
    - All fields are optional to avoid breaking existing configs
    - Extras allowed for custom user-defined path names
    - Automatic path normalization via SecurePath for declared fields and extras
    - Canonical runtime representation uses normalized POSIX-style strings

    Args:
        output_dir: Output directory for results
        checkpoint_path: Path to model checkpoint file
        data_dir: Directory containing datasets
        weights_path: Path to model weights file
        config_path: Path to additional config files

    Extra fields are permitted for custom paths.
    """

    model_config = SettingsConfigDict(
        extra="allow",  # Allow arbitrary user-defined path fields
        validate_default=True,
        validate_assignment=True,
        case_sensitive=True,
        frozen=False,
    )

    # Common standardized path fields (all optional)
    output_dir: SecurePath | None = Field(default=None, description="Output directory for results")

    checkpoint_path: SecurePath | None = Field(
        default=None, description="Path to model checkpoint file"
    )

    data_dir: SecurePath | None = Field(default=None, description="Directory containing datasets")

    weights_path: SecurePath | None = Field(default=None, description="Path to model weights file")

    config_path: SecurePath | None = Field(
        default=None, description="Path to additional config files"
    )

    @model_validator(mode="after")
    def normalize_extra_paths(self) -> Self:
        """Normalize extra PATHS values with the same SecurePath contract.

        Declared fields are validated by Pydantic field annotations. Extras have
        no field annotation, so normalize them here to keep the whole PATHS
        section on a single contract: every value must be a path-like string or
        ``None`` and is stored as a normalized POSIX-style string.
        """
        extra = self.model_extra
        if not extra:
            return self

        normalized_extras: dict[str, SecurePath | None] = {}
        for field_name, value in extra.items():
            try:
                normalized_extras[field_name] = _PATH_VALUE_ADAPTER.validate_python(value)
            except ValidationError as exc:
                raise ValueError(
                    f"PATHS.{field_name} must be a path-like string or None; "
                    "use EXTRAS for non-path arbitrary values"
                ) from exc

        extra.clear()
        extra.update(normalized_extras)
        return self

    def get_path(self, field_name: str) -> SecurePath | None:
        """Get a path field by name, supporting both predefined and extra fields.

        Args:
            field_name: Name of the path field to retrieve

        Returns:
            Normalized POSIX-style path string if the field exists and is not
            None, otherwise None

        Example:
            >>> paths = PathsSettings(matrix_path="data/matrix.txt", custom_data="data/custom.txt")
            >>> paths.get_path("matrix_path")  # Returns SecurePath
            >>> paths.get_path("custom_data")  # Returns custom path if it exists
            >>> paths.get_path("nonexistent")  # Returns None
        """
        if field_name in type(self).model_fields:
            return getattr(self, field_name)

        extra = self.model_extra or {}
        return extra.get(field_name)

    def has_path(self, field_name: str) -> bool:
        """Check if a path field exists and is not None.

        Args:
            field_name: Name of the path field to check

        Returns:
            True if the field exists and is not None
        """
        path = self.get_path(field_name)
        return path is not None
