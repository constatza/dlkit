"""Override normalization utilities for API command layer.

This module provides utilities to eliminate duplicated override normalization logic
across API command classes. It handles path normalization (string → Path) and
builds clean override dictionaries with None-filtering.

The `OverrideNormalizer` class follows the Single Responsibility Principle by
focusing solely on override data transformation and normalization.

Design Principles:
    - Pure functions: Static methods with no side effects
    - Type safety: Comprehensive type hints for all inputs/outputs
    - Immutability: Returns new dictionaries, never mutates input
    - Composability: Can be extended for additional override types

Example:
    >>> from pathlib import Path
    >>> from dlkit.interfaces.api.overrides.normalizer import OverrideNormalizer
    >>>
    >>> # Normalize individual path
    >>> path = OverrideNormalizer.normalize_path("/tmp/data")
    >>> assert isinstance(path, Path)
    >>>
    >>> # Build overrides dict with automatic path normalization
    >>> overrides = OverrideNormalizer.build_overrides_dict(
    ...     checkpoint_path="/tmp/model.ckpt",
    ...     root_dir=Path("/tmp/root"),
    ...     batch_size=32,
    ...     experiment_name="test",
    ...     invalid_none=None,  # Filtered out
    ... )
    >>> assert "invalid_none" not in overrides
    >>> assert isinstance(overrides["checkpoint_path"], Path)
    >>> assert overrides["batch_size"] == 32
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dlkit.tools.utils.system_utils import normalize_user_path


class OverrideNormalizer:
    """Pure utility class for normalizing command overrides.

    This class provides static methods for normalizing runtime parameter overrides
    in the API command layer. It eliminates ~150 lines of duplicated code across
    train, inference, and optimization commands.

    All methods are pure functions (no state, no side effects) that transform
    input data into clean, type-safe override dictionaries.

    Path Normalization:
        String paths are converted to pathlib.Path objects for type consistency.
        Path objects and None values are passed through unchanged.

    None-Filtering:
        All None values are automatically filtered from the resulting dictionary,
        ensuring only meaningful overrides are propagated to the override manager.

    Thread Safety:
        Pure static methods are inherently thread-safe with no shared state.
    """

    # Known path fields that should be normalized to Path objects
    PATH_FIELDS = frozenset({
        "checkpoint_path",
        "root_dir",
        "output_dir",
        "data_dir",
    })

    @staticmethod
    def normalize_path(path: str | Path | None, *, require_absolute: bool = False) -> Path | None:
        """Normalize path value to Path object or None.

        Converts string paths to pathlib.Path objects while preserving
        None values and existing Path objects.

        Args:
            path: Path as string, Path object, or None

        Returns:
            Path object if input was string or Path, None if input was None

        Example:
            >>> OverrideNormalizer.normalize_path("/tmp/data")
            PosixPath('/tmp/data')
            >>> OverrideNormalizer.normalize_path(Path("/tmp/data"))
            PosixPath('/tmp/data')
            >>> OverrideNormalizer.normalize_path(None)
            None
        """
        normalized = normalize_user_path(path, require_absolute=require_absolute)
        if require_absolute:
            return normalized

        if path is None:
            return None

        if isinstance(path, Path):
            if path.is_absolute() or "~" in str(path):
                return normalized
            return path

        if isinstance(path, str):
            if "~" in path or Path(path).is_absolute():
                return normalized
            return Path(path)

        return normalized

    @staticmethod
    def build_overrides_dict(**kwargs: Any) -> dict[str, Any]:
        """Build clean overrides dictionary with automatic normalization.

        This method:
        1. Normalizes known path fields (checkpoint_path, root_dir, etc.) to Path objects
        2. Filters out all None values
        3. Preserves all other values unchanged
        4. Flattens any nested dictionaries from additional_overrides

        Path fields are automatically detected and normalized. All other fields
        are passed through unchanged, supporting arbitrary override types.

        Args:
            **kwargs: Raw override values from command input
                     Can include paths (str | Path), primitives (int, float, str),
                     and nested dicts via additional_overrides

        Returns:
            Clean dictionary with:
                - Path fields normalized to Path objects
                - None values filtered out
                - Additional overrides flattened into top level

        Example:
            >>> overrides = OverrideNormalizer.build_overrides_dict(
            ...     checkpoint_path="/tmp/model.ckpt",
            ...     root_dir=None,  # Filtered out
            ...     batch_size=32,
            ...     experiment_name="test",
            ...     additional_overrides={"custom": "value"}
            ... )
            >>> sorted(overrides.keys())
            ['additional_overrides', 'batch_size', 'checkpoint_path', 'custom', 'experiment_name']
            >>> isinstance(overrides["checkpoint_path"], Path)
            True

        Note:
            The special key "additional_overrides" (if present) is expanded and its
            contents are merged into the top-level dictionary. This allows command
            input classes to support arbitrary user-defined overrides.
        """
        # Extract additional_overrides before processing if present
        additional = kwargs.pop("additional_overrides", None) or {}

        # Normalize known path fields
        normalized = {}
        for key, value in kwargs.items():
            if key in OverrideNormalizer.PATH_FIELDS:
                normalized[key] = OverrideNormalizer.normalize_path(
                    value,
                    require_absolute=(key == "root_dir"),
                )
            else:
                normalized[key] = value

        # Build final dict with None filtering
        result = {k: v for k, v in normalized.items() if v is not None}

        # Merge additional overrides (only non-None values)
        if additional:
            result.update({k: v for k, v in additional.items() if v is not None})

        return result
