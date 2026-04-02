"""Transform list construction for both training and inference contexts.

Kept in adapters.lightning so that runtime.predictor can import it without
depending on the heavier runtime.workflows.factories layer.
"""

from __future__ import annotations

from typing import Any

from torch.nn import ModuleList

from dlkit.tools.config.core.context import BuildContext
from dlkit.tools.config.core.factories import FactoryProvider
from dlkit.tools.config.transform_settings import TransformSettings

_TRANSFORM_MODULE = "dlkit.domain.transforms"


def _apply_transform_default(settings: Any) -> Any:
    """Apply the runtime-owned module-path default for transform settings.

    Args:
        settings: A settings object, typically a TransformSettings subclass.

    Returns:
        Settings with module_path filled in when not already set.
    """
    if getattr(settings, "module_path", None):
        return settings
    if isinstance(settings, TransformSettings):
        model_copy = getattr(settings, "model_copy", None)
        if callable(model_copy):
            return model_copy(update={"module_path": _TRANSFORM_MODULE})
    return settings


def build_transform_list(
    transform_seq: Any,
    shape_spec: Any = None,
    entry_name: str | None = None,
    validate_execution: bool = False,
) -> tuple[ModuleList, tuple[int, ...] | None]:
    """Instantiate transforms with analytical shape inference.

    Args:
        transform_seq: Sequence of TransformSettings.
        shape_spec: Optional shape specification for pre-allocation.
        entry_name: Entry name for shape lookup.
        validate_execution: Whether to validate with dummy tensors.

    Returns:
        Tuple of (ModuleList, inferred_output_shape | None).
    """
    import torch

    current_shape = None
    if shape_spec and entry_name:
        current_shape = shape_spec.get_shape(entry_name)

    module_list = ModuleList()
    dummy_input = None
    if validate_execution and current_shape is not None:
        dummy_input = torch.zeros(current_shape)

    for transform_settings in transform_seq:
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
