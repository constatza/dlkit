"""Build context for dependency injection in settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BuildContext(BaseModel):
    """Context object for passing dependencies during object construction.

    This replaces the need for complex parameter passing in the old build() methods.
    It provides a clean way to inject dependencies and environment information.

    Args:
        mode: The current execution mode (training, inference, etc.)
        device: The target device for computation
        random_seed: Random seed for reproducibility
        working_directory: Current working directory
        checkpoint_path: Path to model checkpoint if needed
        overrides: Additional keyword arguments for construction
    """

    mode: str = Field(description="Execution mode")
    device: str = Field(default="auto", description="Target device")
    random_seed: int | None = Field(default=None, description="Random seed")
    working_directory: Path = Field(
        default_factory=lambda: Path.cwd(), description="Working directory"
    )
    checkpoint_path: Path | None = Field(default=None, description="Checkpoint path")
    overrides: dict[str, Any] = Field(default_factory=dict, description="Additional overrides")

    # Pydantic v2 configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def with_overrides(self, **kwargs) -> BuildContext:
        """Create a new context with additional overrides.

        Args:
            **kwargs: Additional overrides to merge

        Returns:
            BuildContext: New context with merged overrides
        """
        new_overrides = {**self.overrides, **kwargs}
        return self.model_copy(update={"overrides": new_overrides})

    def get_override(self, key: str, default: Any = None) -> Any:
        """Get an override value by key.

        Args:
            key: The override key
            default: Default value if key not found

        Returns:
            Any: The override value or default
        """
        return self.overrides.get(key, default)
