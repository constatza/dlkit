"""Type guards for CLI commands.

Provides TypeGuard functions to safely narrow types at runtime
without relying on isinstance() checks against Protocol objects.
"""

from __future__ import annotations

from typing import Any, TypeGuard

from dlkit.infrastructure.config.protocols import TrainingSettingsProtocol


def is_training_settings(obj: Any) -> TypeGuard[TrainingSettingsProtocol]:
    """Check if an object conforms to TrainingSettingsProtocol.

    Args:
        obj: Object to check.

    Returns:
        True if obj has the required TrainingSettingsProtocol attributes.
    """
    return (
        hasattr(obj, "TRAINING")
        and hasattr(obj, "patch")
        and hasattr(obj, "to_dict")
        and callable(getattr(obj, "patch", None))
    )
