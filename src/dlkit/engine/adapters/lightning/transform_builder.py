"""Transform list construction for both training and inference contexts.

Kept in adapters.lightning so that runtime.predictor can import it without
depending on the heavier runtime.workflows.factories layer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from torch.nn import ModuleList

from dlkit.infrastructure.config.core.context import BuildContext
from dlkit.infrastructure.config.core.factories import FactoryProvider
from dlkit.infrastructure.config.transform_settings import TransformSettings

_TRANSFORM_MODULE = "dlkit.domain.transforms"


def _strip_serialized_default_fields(spec: Mapping[object, object]) -> dict[object, object]:
    """Drop serialized default fields to match typed-settings construction.

    Checkpoint metadata serializes transform settings with default-valued fields
    materialized. The typed settings path passes only explicitly set init kwargs,
    so normalize serialized mappings back to that effective contract.

    Args:
        spec: Serialized transform mapping.

    Returns:
        Copy of ``spec`` with default-valued optional fields removed.
    """
    normalized = dict(spec)
    for field_name, field_info in TransformSettings.model_fields.items():
        if field_name == "name" or field_name not in normalized:
            continue
        if field_info.is_required():
            continue
        default = field_info.get_default(call_default_factory=True)
        if normalized[field_name] == default:
            normalized.pop(field_name)
    return normalized


def _normalize_transform_spec(spec: object) -> TransformSettings:
    """Normalize one transform specification into typed settings.

    Args:
        spec: Transform configuration provided either as typed settings or a
            serialized mapping compatible with ``TransformSettings``.

    Returns:
        Validated ``TransformSettings`` instance.

    Raises:
        TypeError: If the spec is neither typed settings nor a mapping.
    """
    if isinstance(spec, TransformSettings):
        return spec
    if isinstance(spec, Mapping):
        mapping_spec = cast("Mapping[object, object]", spec)
        return TransformSettings.model_validate(_strip_serialized_default_fields(mapping_spec))
    raise TypeError(
        "Transform specifications must be TransformSettings instances or mappings "
        f"compatible with TransformSettings, got {type(spec).__name__}."
    )


def _normalize_transform_specs(transform_seq: Sequence[object]) -> tuple[TransformSettings, ...]:
    """Normalize a transform sequence into typed settings.

    Args:
        transform_seq: Sequence of transform specifications.

    Returns:
        Tuple of validated ``TransformSettings`` instances.
    """
    return tuple(_normalize_transform_spec(spec) for spec in transform_seq)


def _apply_transform_default(settings: TransformSettings) -> TransformSettings:
    """Apply the runtime-owned module-path default for transform settings.

    Args:
        settings: Typed transform settings.

    Returns:
        Settings with module_path filled in when not already set.
    """
    if settings.module_path:
        return settings
    model_copy = getattr(settings, "model_copy", None)
    if callable(model_copy):
        return model_copy(update={"module_path": _TRANSFORM_MODULE})
    return settings


def build_transform_list(
    transform_seq: Sequence[object],
    entry_name: str | None = None,
    validate_execution: bool = False,
) -> tuple[ModuleList, tuple[int, ...] | None]:
    """Instantiate transforms with analytical shape inference.

    Args:
        transform_seq: Sequence of transform specifications accepted as either
            ``TransformSettings`` instances or serialized mappings compatible
            with ``TransformSettings``.
        entry_name: Entry name for shape lookup.
        validate_execution: Whether to validate with dummy tensors.

    Returns:
        Tuple of (ModuleList, inferred_output_shape | None).
    """
    import torch

    current_shape = None
    normalized_settings = _normalize_transform_specs(transform_seq)

    module_list = ModuleList()
    dummy_input = None
    if validate_execution and current_shape is not None:
        dummy_input = torch.zeros(current_shape)

    for transform_settings in normalized_settings:
        context = BuildContext(mode="transform_chain")
        module = FactoryProvider.create_component(
            _apply_transform_default(transform_settings),
            context,
        )

        if current_shape is not None and hasattr(module, "infer_output_shape"):
            current_shape = module.infer_output_shape(current_shape)

        if validate_execution and dummy_input is not None:
            dummy_input = module(dummy_input)
            if current_shape is not None:
                assert tuple(dummy_input.shape) == current_shape

        module_list.append(module)

    return module_list, current_shape
