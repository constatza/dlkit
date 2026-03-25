"""Settings updater with immutable semantics.

This module provides ``update_settings()`` as a thin wrapper around
``patch_model()`` for backward compatibility.  All config classes are now
frozen (``frozen=True``); updates produce new instances instead of mutating
in place.

Common Use Cases
----------------
1. Updating optimizer learning rate::

       from dlkit.tools.config import load_settings, update_settings

       config = load_settings("config.toml")
       config = update_settings(config, {"TRAINING": {"optimizer": {"lr": 0.001}}})

2. Injecting in-memory data::

       import numpy as np
       from dlkit.tools.config.data_entries import Feature

       config = update_settings(
           config.DATASET,
           {"features": (Feature(name="x", value=np.random.randn(1000, 20)),)},
       )

Technical Details
-----------------
- Returns a **new** object (the original is unchanged).
- Supports dotted keys: ``{"TRAINING.optimizer.lr": 0.001}`` expands correctly.
- Strict merge semantics: key collisions raise ``ValueError`` (no silent overwrites).
- Type-validated: each patch value is validated against the field annotation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dlkit.tools.config.core.patching import patch_model

if TYPE_CHECKING:
    from .base_settings import BasicSettings


def update_settings[T: "BasicSettings"](
    settings: T,
    updates: dict[str, Any],
    validate: bool = True,
) -> T:
    """Return a new settings instance with *updates* applied.

    Delegates to :func:`~dlkit.tools.config.core.patching.patch_model` which
    compiles mixed overrides (plain nested dicts and dotted keys), validates
    each patch value against its field annotation, and produces a new frozen
    model instance.

    The original *settings* object is **never mutated**.

    Args:
        settings: Settings instance to update.
        updates: Nested dict of updates.  Dotted keys like ``"a.b"`` are
            expanded to ``{"a": {"b": ...}}``.
        validate: Ignored (kept for API compatibility; validation always
            occurs via Pydantic field annotations).

    Returns:
        T: New settings instance (same concrete type) with updates applied.

    Raises:
        ValueError: On key conflicts in mixed overrides.
        KeyError: Unknown field name.
        pydantic.ValidationError: On type mismatches.

    Examples:
        Update a nested field::

            new_cfg = update_settings(cfg, {"TRAINING": {"epochs": 100}})
            assert new_cfg.TRAINING.epochs == 100
            assert cfg.TRAINING.epochs != 100  # original unchanged

        Dotted-key shorthand::

            new_cfg = update_settings(cfg, {"TRAINING.optimizer.lr": 0.001})
    """
    return patch_model(settings, updates, revalidate=validate)
