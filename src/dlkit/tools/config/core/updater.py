"""Settings updater with deep merge and in-place mutation.

This module provides the core `update_settings()` function for mutating nested
Pydantic settings in-place. The mutable settings architecture (frozen=False with
validate_assignment=True) was chosen to prevent data loss with excluded fields.

Architecture Rationale
----------------------
DLKit uses mutable settings (frozen=False) instead of immutable settings to solve
a critical issue with value-based data entries:

**Problem with Immutable (frozen=True):**
- ValueFeature.value is marked with exclude=True (not serialized)
- model_copy() requires serialization → deserialization
- Excluded fields are LOST during this cycle
- Result: In-memory arrays disappear after updates

**Solution with Mutable (frozen=False):**
- Direct attribute mutation (no serialization)
- Object identity preserved across updates
- Excluded fields (like ValueFeature.value) remain intact
- validate_assignment=True ensures type safety on every setattr()

Common Use Cases
----------------
1. Adding features to existing dataset configuration:
   >>> from dlkit.tools.config import load_settings
   >>> from dlkit.tools.config.data_entries import Feature
   >>> config = load_settings("config.toml")
   >>> new_feature = Feature(name="z", path="data/z.npy")
   >>> update_settings(
   ...     config.DATASET,
   ...     {
   ...         "features": config.DATASET.features
   ...         + [
   ...             new_feature,
   ...         ]
   ...     },
   ... )

2. Updating optimizer learning rate:
   >>> update_settings(config, {"TRAINING": {"optimizer": {"lr": 0.001}}})

3. Injecting in-memory data:
   >>> import numpy as np
   >>> features = np.random.randn(1000, 20).astype(np.float32)
   >>> update_settings(config.DATASET, {"features": (Feature(name="x", value=features),)})

Technical Details
-----------------
- Returns the SAME object (mutated in-place)
- Deep merges nested dictionaries
- Recursively updates nested BaseModel instances
- Validation happens automatically via validate_assignment=True
- No serialization overhead
"""

from __future__ import annotations
from typing import Any, TYPE_CHECKING
from pydantic import BaseModel

if TYPE_CHECKING:
    from .base_settings import BasicSettings


def update_settings[T: BasicSettings](
    settings: T,
    updates: dict[str, Any],
    validate: bool = True,
) -> T:
    """Update settings by direct mutation - NO serialization.

    This function mutates the settings in-place, preserving object identity.
    It correctly handles:
    - Nested Pydantic models: recursively updates in-place
    - Plain dicts (like EXTRAS): overwrites completely
    - All other types: overwrites via setattr

    Because settings now have frozen=False, direct mutation is allowed.
    validate_assignment=True ensures type safety on each setattr.

    Args:
        settings: Settings instance to mutate in-place
        updates: Nested dict of updates (only specified fields are overwritten)
        validate: Ignored (validation happens automatically via validate_assignment)

    Returns:
        The same settings instance (mutated in-place)

    Examples:

        Update multiple sections at once:
        >>> update_settings(
        ...     settings,
        ...     {
        ...         "TRAINING": {"epochs": 100},
        ...         "MLFLOW": {"server": {"host": "localhost"}},
        ...     },
        ... )

        Settings are mutated in-place (same object returned):
        >>> orig_id = id(settings)
        >>> new_settings = update_settings(settings, {"TRAINING": {"epochs": 100}})
        >>> assert id(new_settings) == orig_id
    """
    # Recursively apply updates by direct attribute assignment
    for key, new_value in updates.items():
        # Guard clause: Check if field exists (or if model allows extras)
        allows_extras = False
        if hasattr(settings, "model_config"):
            allows_extras = settings.model_config.get("extra") == "allow"

        if not hasattr(settings, key) and not allows_extras:
            raise ValueError(f"Unknown setting: {key}")

        old_value = getattr(settings, key, None)

        # Handle different value types with pattern matching
        match (new_value, old_value):
            # BaseModel update on BaseModel field - convert to dict and recurse
            case (BaseModel(), BaseModel()):
                value_dict = new_value.model_dump(exclude_unset=True)
                update_settings(old_value, value_dict, validate=validate)

            # BaseModel update on non-BaseModel field - replace
            case (BaseModel(), _):
                setattr(settings, key, new_value)

            # Dict update on BaseModel field - recurse
            case (dict(), BaseModel()):
                update_settings(old_value, new_value, validate=validate)

            # Dict update on dict field - merge
            case (dict(), dict()):
                merged_dict = _merge_plain_dict(old_value, new_value)
                setattr(settings, key, merged_dict)

            # Dict on None (new field with extra="allow") - set
            case (dict(), None):
                setattr(settings, key, new_value)

            # Any other case - direct assignment
            case _:
                setattr(settings, key, new_value)

    return settings


def _merge_plain_dict(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Merge two plain dictionaries while preserving existing keys."""

    merged = base.copy()
    for key, value in updates.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge_plain_dict(current, value)
        else:
            merged[key] = value
    return merged
