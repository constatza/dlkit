"""Compatibility shim: re-exports SessionSettings stub from workflow_settings_base.

Will be removed in Task 5 when all callers are updated.
"""

from dlkit.infrastructure.config.workflow_settings_base import SessionSettings

__all__ = ["SessionSettings"]
