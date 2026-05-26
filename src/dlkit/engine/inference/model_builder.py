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
    from dlkit.domain.nn.contracts import ModelContractSpec

logger = get_logger(__name__)

# Keys that contract variants inject into from_contract(**kwargs). Stripping them
# from hyperparams prevents "got multiple values for keyword argument" errors when
# both the checkpoint's init_kwargs and the contract supply the same parameter.
_CONTRACT_KEYS: frozenset[str] = frozenset(
    {
        "in_features",
        "out_features",
        "in_shape",
        "out_shape",
        "in_channels",
        "out_channels",
        "spatial_shape",
        "seq_len",
        "edge_dim",
        "branch_shape",
        "query_shape",
    }
)


def _strip_contract_keys(hyperparams: dict[str, Any]) -> dict[str, Any]:
    """Remove shape keys that a ModelContractSpec will supply via from_contract().

    Args:
        hyperparams: Raw init kwargs extracted from checkpoint model settings.

    Returns:
        Filtered kwargs with contract-owned keys removed.
    """
    return {k: v for k, v in hyperparams.items() if k not in _CONTRACT_KEYS}


def build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    contract: ModelContractSpec | None = None,
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
    if contract is not None:
        hyperparams = _strip_contract_keys(hyperparams)
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
