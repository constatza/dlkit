"""Single source of truth for ModelComponentSettings factory-directive fields.

Factory directives (activation, normalize) are excluded from raw init kwargs
(``dlkit_init_kwarg: False`` in ModelComponentSettings) and instead resolved
per-model via ``model_accepts_kwarg``. Centralizing the field list and merge
logic here keeps model construction, checkpoint serialization, and checkpoint
reconstruction from drifting out of sync with each other.
"""

from __future__ import annotations

from typing import Any

from dlkit.domain.nn.factory import model_accepts_kwarg
from dlkit.infrastructure.config.model_components import ModelComponentSettings

FACTORY_DIRECTIVE_FIELDS: tuple[str, ...] = ("activation", "normalize")


def resolve_factory_kwargs(
    model_settings: ModelComponentSettings,
    model_cls: type | None,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Merge factory-directive settings fields into kwargs for accepting models.

    Args:
        model_settings: Settings holding the factory-directive field values.
        model_cls: Model class to check acceptance against, or None if unresolved.
        kwargs: Existing kwargs/hyper_kwargs to merge into (not mutated).

    Returns:
        A new dict with factory directives added where the model class accepts them.
    """
    if model_cls is None:
        return kwargs
    resolved = dict(kwargs)
    for field_name in FACTORY_DIRECTIVE_FIELDS:
        value = getattr(model_settings, field_name, None)
        if (
            value is not None
            and field_name not in resolved
            and model_accepts_kwarg(model_cls, field_name)
        ):
            resolved[field_name] = value
    return resolved
