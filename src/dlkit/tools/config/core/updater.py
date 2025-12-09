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
   >>> update_settings(config.DATASET, {
   ...     "features": config.DATASET.features + (new_feature,)
   ... })

2. Updating optimizer learning rate:
   >>> update_settings(config, {
   ...     "TRAINING": {"optimizer": {"lr": 0.001}}
   ... })

3. Injecting in-memory data:
   >>> import numpy as np
   >>> features = np.random.randn(1000, 20).astype(np.float32)
   >>> update_settings(config.DATASET, {
   ...     "features": (Feature(name="x", value=features),)
   ... })

Technical Details
-----------------
- Returns the SAME object (mutated in-place)
- Deep merges nested dictionaries
- Recursively updates nested BaseModel instances
- Validation happens automatically via validate_assignment=True
- No serialization overhead
"""

from __future__ import annotations

from typing import Any, TypeVar, Union, cast, get_args, get_origin
from types import UnionType

from pydantic import BaseModel

from .base_settings import BasicSettings

T = TypeVar("T", bound=BaseModel)

_EMPTY = object()


def update_settings(
    settings: BasicSettings,
    updates: dict[str, Any],
    validate: bool = True,
) -> BasicSettings:
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
        Update nested fields without losing other settings:
        >>> update_settings(settings, {
        ...     "DATASET": {
        ...         "features": [Feature(name="x", path="data.npy")]
        ...     }
        ... })

        Update multiple sections at once:
        >>> update_settings(settings, {
        ...     "TRAINING": {"epochs": 100},
        ...     "MLFLOW": {"server": {"host": "localhost"}},
        ... })

        Settings are mutated in-place (same object returned):
        >>> orig_id = id(settings)
        >>> new_settings = update_settings(settings, {"TRAINING": {"epochs": 100}})
        >>> assert id(new_settings) == orig_id
    """
    # Recursively apply updates by direct attribute assignment
    for key, value in updates.items():
        # Guard clause: Check if field exists (or if model allows extras)
        has_field = hasattr(settings, key)
        allows_extras = False
        if hasattr(settings, 'model_config'):
            config = settings.model_config
            allows_extras = (isinstance(config, dict) and config.get('extra') == 'allow')

        if not has_field and not allows_extras:
            raise ValueError(f"Unknown setting: {key}")

        current = getattr(settings, key, None)

        # Handle different value types with pattern matching
        match (value, current):
            # BaseModel update on BaseModel field - convert to dict and recurse
            case (BaseModel(), BaseModel()):
                value_dict = value.model_dump(exclude_unset=True)
                update_settings(current, value_dict, validate=validate)

            # BaseModel update on non-BaseModel field - replace
            case (BaseModel(), _):
                setattr(settings, key, value)

            # Dict update on BaseModel field - recurse
            case (dict(), BaseModel()):
                update_settings(current, value, validate=validate)

            # Dict update on dict field - merge
            case (dict(), dict()):
                merged_dict = _merge_plain_dict(current, value)
                setattr(settings, key, merged_dict)

            # Dict on None (new field with extra="allow") - set
            case (dict(), None):
                setattr(settings, key, value)

            # Any other case - direct assignment
            case _:
                setattr(settings, key, value)

    return settings


def _deep_merge(
    base: dict[str, Any],
    updates: dict[str, Any],
    original_settings: BaseModel,
) -> dict[str, Any]:
    """Deep merge updates into base dict, checking types from original settings.

    Strategy:
    - Check the type of each field in the ORIGINAL settings object
    - If field is a BaseModel instance with specific known fields → recurse
    - If field is a plain dict → merge existing keys with new ones
    - All other types → overwrite

    Args:
        base: Current state as dict
        updates: Updates to apply
        original_settings: Original settings object for type checking

    Returns:
        Merged dict with updates applied
    """
    result = base.copy()

    for key, raw_update_value in updates.items():
        update_value = _normalize_update_value(raw_update_value)

        # Nothing explicit to update (e.g. empty BaseModel payload)
        if update_value is _EMPTY:
            continue

        # Guard against BaseModel producing empty dict
        if isinstance(update_value, dict) and not update_value:
            continue

        base_value = result.get(key)

        # Both are dicts - check the type in the ORIGINAL settings
        if isinstance(base_value, dict) and isinstance(update_value, dict):
            # Get the actual field value from original settings
            original_field_value = getattr(original_settings, key, None)

            # Check if original field is a Pydantic model instance
            if isinstance(original_field_value, BaseModel):
                # It's a structured Pydantic model - RECURSE to preserve sub-fields
                result[key] = _deep_merge(
                    base_value,
                    update_value,
                    original_field_value,  # Pass nested model as context
                )
            else:
                # Plain dict - merge without deleting unspecified keys
                result[key] = _merge_plain_dict(base_value, update_value)
        else:
            # All other types - OVERWRITE
            # Includes: list, tuple, str, int, float, bool, Path, None, etc.
            result[key] = update_value

    return result


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


def _construct_with_nested_models(
    model_class: type[T],
    data: dict[str, Any],
) -> T:
    """Construct Pydantic model with proper nested model reconstruction.

    This function ensures that nested dicts are converted to their proper
    Pydantic model types when using model_construct as a fallback. This
    prevents AttributeError when accessing methods on nested models that
    ended up as plain dicts.

    Args:
        model_class: Pydantic model class to construct.
        data: Dictionary data to construct from.

    Returns:
        Model instance with all nested models properly constructed.
    """
    processed_data: dict[str, Any] = {}

    for key, value in data.items():
        field_info = model_class.model_fields.get(key)
        if field_info is None:
            # Extra field - pass through as-is
            processed_data[key] = value
            continue

        # Determine the expected field type
        field_type = field_info.annotation

        # Handle Optional[SomeModel], Union[SomeModel, None], etc.
        origin = get_origin(field_type)
        if origin in (Union, UnionType):
            args = get_args(field_type)
            field_type = next((arg for arg in args if arg is not type(None)), field_type)

        # Recursively construct nested Pydantic models
        if isinstance(value, dict) and isinstance(field_type, type) and issubclass(field_type, BaseModel):
            processed_data[key] = _construct_with_nested_models(field_type, value)
        elif isinstance(value, (list, tuple)):
            # Handle lists/tuples of nested models
            processed_items = []
            item_type = field_type
            item_origin = get_origin(field_type)
            if item_origin in (tuple, list):
                item_args = get_args(field_type)
                if item_args:
                    item_type = item_args[0]

            for item in value:
                if isinstance(item, dict) and isinstance(item_type, type) and issubclass(item_type, BaseModel):
                    processed_items.append(_construct_with_nested_models(item_type, item))
                else:
                    processed_items.append(item)

            # Preserve original container type
            if isinstance(value, tuple):
                processed_data[key] = tuple(processed_items)
            else:
                processed_data[key] = processed_items
        else:
            processed_data[key] = value

    return model_class.model_construct(**processed_data)


def _normalize_update_value(value: Any) -> Any:
    """Convert BaseModel updates into partial dictionaries for safe merging.

    This ensures that all nested Pydantic models are serialized to dicts,
    allowing proper re-validation with context when model_validate is called.
    """
    if isinstance(value, BaseModel):
        # Only materialise explicitly set fields to avoid clobbering defaults
        data = value.model_dump(exclude_unset=True)
        if not data:
            return _EMPTY
        return {k: _normalize_update_value(v) for k, v in data.items()}

    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, nested_value in value.items():
            normalized_value = _normalize_update_value(nested_value)
            if normalized_value is _EMPTY:
                continue
            normalized[key] = normalized_value
        return normalized

    if isinstance(value, (list, tuple)):
        # Recursively normalize items in sequences
        # This ensures BaseModel instances inside tuples/lists are converted to dicts
        normalized_items = [_normalize_update_value(item) for item in value]
        # Filter out _EMPTY items
        normalized_items = [item for item in normalized_items if item is not _EMPTY]
        # Preserve original container type
        if isinstance(value, tuple):
            return tuple(normalized_items)
        return normalized_items

    return value
