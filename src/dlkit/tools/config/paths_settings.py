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

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from dlkit.core.datatypes.secure_uris import SecurePath
from .core.base_settings import BasicSettings


class PathsSettings(BasicSettings):
    """Optional paths configuration with automatic resolution and extras support.

    Provides standardized path fields with DLKit's automatic path resolution.
    All relative paths are resolved against SESSION.root_dir following DLKit's
    standard path resolution hierarchy.

    Features:
    - All fields are optional to avoid breaking existing configs
    - Extras allowed for custom user-defined paths
    - Automatic path resolution via SecurePath
    - Type safety for predefined common fields

    Args:
        matrix_path: Path to matrix data file
        rhs_path: Path to right-hand side vector file
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
    matrix_path: SecurePath | None = Field(
        default=None,
        description="Path to matrix data file"
    )

    rhs_path: SecurePath | None = Field(
        default=None,
        description="Path to right-hand side vector file"
    )

    output_dir: SecurePath | None = Field(
        default=None,
        description="Output directory for results"
    )

    checkpoint_path: SecurePath | None = Field(
        default=None,
        description="Path to model checkpoint file"
    )

    data_dir: SecurePath | None = Field(
        default=None,
        description="Directory containing datasets"
    )

    weights_path: SecurePath | None = Field(
        default=None,
        description="Path to model weights file"
    )

    config_path: SecurePath | None = Field(
        default=None,
        description="Path to additional config files"
    )

    def get_path(self, field_name: str) -> SecurePath | None:
        """Get a path field by name, supporting both predefined and extra fields.

        Args:
            field_name: Name of the path field to retrieve

        Returns:
            SecurePath if the field exists and is not None, otherwise None

        Example:
            >>> paths = PathsSettings(matrix_path="data/matrix.txt", custom_data="data/custom.txt")
            >>> paths.get_path("matrix_path")  # Returns SecurePath
            >>> paths.get_path("custom_data")   # Returns custom path if it exists
            >>> paths.get_path("nonexistent")  # Returns None
        """
        return getattr(self, field_name, None)

    def has_path(self, field_name: str) -> bool:
        """Check if a path field exists and is not None.

        Args:
            field_name: Name of the path field to check

        Returns:
            True if the field exists and is not None
        """
        path = self.get_path(field_name)
        return path is not None