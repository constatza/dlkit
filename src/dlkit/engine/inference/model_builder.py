"""Model construction from checkpoint payloads."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, cast

import torch

from dlkit.common.errors import WorkflowError
from dlkit.domain.nn.factory import build_model as _build_model
from dlkit.infrastructure.config.model_components import extract_init_kwargs
from dlkit.infrastructure.utils.logging_config import get_logger

from .checkpoint_reader import detect_checkpoint_dtype, extract_model_settings

if TYPE_CHECKING:
    from dlkit.common.shapes import ShapeSummary

logger = get_logger(__name__)


def _shape_spec_to_contract(shape_spec: Any) -> Any:
    """Convert a ShapeSummary to a ModelContractSpec.

    Temporary bridge until Steps 7-9 replace this with GeometrySpec → resolve_contract().
    """
    if shape_spec is None:
        return None
    from dlkit.domain.nn.contracts import BranchTrunkSpec, GridOperatorSpec, TabulaRSpec

    try:
        in_shapes = shape_spec.in_shapes
        out_shapes = shape_spec.out_shapes
        if len(in_shapes) >= 2:
            return BranchTrunkSpec(
                branch_shape=in_shapes[0],
                query_shape=in_shapes[1],
                out_features=out_shapes[0][0],
            )
        in_shape = in_shapes[0]
        out_shape = out_shapes[0]
        if len(in_shape) == 1:
            return TabulaRSpec(in_shape=in_shape, out_shape=out_shape)
        return GridOperatorSpec(
            in_channels=in_shape[0],
            out_channels=out_shape[0],
            spatial_shape=in_shape[1:],
        )
    except AttributeError, IndexError, TypeError:
        return None


def build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    shape_spec: ShapeSummary | None = None,
) -> torch.nn.Module:
    """Instantiate and load a model from checkpoint metadata and weights."""
    model_settings = extract_model_settings(checkpoint)
    raw_state_dict = checkpoint.get("state_dict", checkpoint)
    if isinstance(raw_state_dict, dict) and any(key.startswith("model.") for key in raw_state_dict):
        state_dict = {
            key[len("model.") :]: value
            for key, value in raw_state_dict.items()
            if key.startswith("model.")
        }
    else:
        state_dict = raw_state_dict if isinstance(raw_state_dict, dict) else {}
    checkpoint_dtype = detect_checkpoint_dtype(state_dict)

    if isinstance(model_settings.name, type):
        model_cls = model_settings.name
    else:
        class_name = model_settings.name
        module_path = model_settings.module_path
        if not isinstance(class_name, str) or not module_path:
            raise WorkflowError(
                f"Cannot resolve model class: name={class_name!r}, module_path={module_path!r}"
            )
        try:
            module = importlib.import_module(module_path)
            model_cls = getattr(module, class_name)
        except (ImportError, AttributeError) as exc:
            raise WorkflowError(
                f"Failed to import model class {module_path}.{class_name}: {exc}"
            ) from exc

    hyperparams = extract_init_kwargs(model_settings)
    contract = _shape_spec_to_contract(shape_spec)
    model = _build_model(cast("type[torch.nn.Module]", model_cls), hyperparams, contract=contract)
    logger.debug("Converting model to checkpoint dtype: {}", checkpoint_dtype)
    model = model.to(dtype=checkpoint_dtype)
    try:
        model.load_state_dict(state_dict, strict=True)
        logger.debug("Successfully loaded model weights from checkpoint")
    except RuntimeError as exc:
        raise WorkflowError(
            f"State dict mismatch loading {type(model).__name__}: {exc}. "
            "If intentional partial loading is required, call load_state_dict(strict=False) directly.",
            {"model_type": type(model).__name__},
        ) from exc
    except Exception as exc:
        raise WorkflowError(
            f"Failed to load state dict into model: {exc}",
            {"model_type": type(model).__name__},
        ) from exc
    model.eval()
    return model
