"""Settings updater with deep merge and full validation.

This module provides a simple interface for updating nested Pydantic settings
without manual nested model_copy() calls.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .base_settings import BasicSettings

_EMPTY = object()


def update_settings(
    settings: BasicSettings,
    updates: dict[str, Any],
    validate: bool = True,
) -> BasicSettings:
    """Deep merge updates into settings with full validation.

    This function merges the updates dict into the settings, preserving all
    unspecified fields. It correctly handles:
    - Nested Pydantic models: recursively merges
    - Plain dicts (like EXTRAS): overwrites completely
    - All other types: overwrites

    Args:
        settings: Current settings instance
        updates: Nested dict of updates (only specified fields are overwritten)
        validate: If True, re-validate entire result after merge (default: True)

    Returns:
        New settings instance with updates applied and optionally validated

    Examples:
        Update nested fields without losing other settings:
        >>> new_settings = update_settings(settings, {
        ...     "DATASET": {
        ...         "features": [Feature(name="x", path="data.npy")]
        ...     }
        ... })

        Update multiple sections at once:
        >>> new_settings = update_settings(settings, {
        ...     "TRAINING": {"epochs": 100},
        ...     "MLFLOW": {"server": {"host": "localhost"}},
        ... })

        Skip validation for performance (use with caution):
        >>> new_settings = update_settings(
        ...     settings,
        ...     {"TRAINING": {"epochs": 100}},
        ...     validate=False
        ... )
    """
    # Get current state as dict
    current = settings.model_dump()

    # Merge using the original settings object to check types
    merged = _deep_merge(current, updates, settings)

    # Always use model_validate to ensure proper type coercion of nested models
    # This prevents serialization warnings about dicts instead of proper Pydantic models
    if validate:
        # model_validate forces complete re-validation of all fields
        # All nested Pydantic models are re-instantiated and validated
        return settings.__class__.model_validate(merged)
    else:
        # Skip custom validators but still do type coercion
        # This ensures nested dicts are properly converted to Pydantic models
        # preventing serialization warnings while bypassing field validation
        try:
            # Use model_validate without strict mode - allows type coercion without full validation
            return settings.__class__.model_validate(merged, strict=False)
        except Exception:
            # Fallback to model_construct if model_validate fails
            # This can happen with very incomplete configs during lazy loading
            return settings.__class__.model_construct(**merged)


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


def _normalize_update_value(value: Any) -> Any:
    """Convert BaseModel updates into partial dictionaries for safe merging."""

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

    return value
