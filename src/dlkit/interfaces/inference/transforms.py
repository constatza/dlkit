"""Transform loading from checkpoints for inference.

Simplified transform loading without unnecessary class wrappers.
Direct functions for loading fitted transforms from checkpoints.
"""

from __future__ import annotations

from typing import Any

import torch
from loguru import logger

from dlkit.core.training.transforms.chain import TransformChain


def load_transforms_from_checkpoint(
    checkpoint: dict[str, Any],
) -> tuple[dict[str, TransformChain], dict[str, TransformChain]]:
    """Load fitted transforms from checkpoint, separated by type.

    Expects the named ModuleDict format produced by NamedBatchTransformer:
    state dict keys ``_batch_transformer._feature_chains.<name>.*`` and
    ``_batch_transformer._target_chains.<name>.*``.

    Args:
        checkpoint: Loaded checkpoint dictionary

    Returns:
        Tuple of (feature_transforms, target_transforms) dictionaries
    """
    feature_transforms: dict[str, TransformChain] = {}
    target_transforms: dict[str, TransformChain] = {}

    state_dict = checkpoint.get("state_dict", {})
    dlkit_metadata = checkpoint.get("dlkit_metadata", {})
    entry_configs = dlkit_metadata.get("entry_configs", [])

    entry_configs_dict = {
        e["name"]: e for e in entry_configs if isinstance(e, dict) and "name" in e
    }

    has_named = any(
        k.startswith("_batch_transformer._feature_chains.")
        or k.startswith("_batch_transformer._target_chains.")
        for k in state_dict.keys()
    )

    if has_named:
        logger.info(
            "Loading transforms from named ModuleDict format "
            "(_batch_transformer._feature_chains / _batch_transformer._target_chains)"
        )
        feature_transforms = _load_named_transforms(
            "_batch_transformer._feature_chains", state_dict, entry_configs_dict
        )
        target_transforms = _load_named_transforms(
            "_batch_transformer._target_chains", state_dict, entry_configs_dict
        )
    else:
        logger.info("No fitted transforms found in checkpoint")

    return feature_transforms, target_transforms


def _load_named_transforms(
    prefix: str,
    state_dict: dict[str, Any],
    entry_configs: dict[str, Any],
) -> dict[str, "TransformChain"]:
    """Load transforms from named ModuleDict format (_batch_transformer._feature_chains.<name>.*).

    This is the current format produced by NamedBatchTransformer. Keys have the
    structure ``<prefix>.<entry_name>.<param_path>``.

    Args:
        prefix: State dict prefix (e.g., ``_batch_transformer._feature_chains``).
        state_dict: Full model state dictionary.
        entry_configs: Entry configurations keyed by name.

    Returns:
        Dictionary mapping entry names to reconstructed TransformChain objects.
    """
    transforms: dict[str, TransformChain] = {}

    # Group by entry name (first component after prefix)
    grouped: dict[str, dict[str, Any]] = {}
    prefix_dot = f"{prefix}."
    for key, value in state_dict.items():
        if not key.startswith(prefix_dot):
            continue
        remainder = key[len(prefix_dot) :]
        parts = remainder.split(".", 1)
        if not parts:
            continue
        entry_name = parts[0]
        sub_key = parts[1] if len(parts) > 1 else ""
        if entry_name not in grouped:
            grouped[entry_name] = {}
        if sub_key:
            grouped[entry_name][sub_key] = value

    for entry_name, chain_state in grouped.items():
        chain = _reconstruct_transform_chain(entry_name, chain_state, entry_configs)
        if chain is not None:
            transforms[entry_name] = chain

    return transforms


def _register_transform_buffer(chain: TransformChain, key: str, state: dict[str, Any]) -> None:
    """Register a buffer in the transform chain.

    Args:
        chain: The transform chain
        key: Dot-separated key path (e.g., 'transforms.0.min')
        state: State dict containing the buffer value
    """
    from torch.nn import ModuleList

    parts = key.split(".")
    module = chain

    # Navigate to the target module
    for part in parts[:-1]:
        if not part.isdigit():
            module = getattr(module, part)
            continue

        # Handle numeric indices for ModuleList or TransformChain
        idx = int(part)
        if isinstance(module, ModuleList):
            module = module[idx]
        elif hasattr(module, "transforms"):
            # TransformChain has transforms attribute
            module = module.transforms[idx]  # type: ignore[index]
        else:
            raise ValueError(f"Cannot index into {type(module)} at {part}")

    # Register the buffer
    buffer_name = parts[-1]
    buffer_value = state[key]
    if not hasattr(module, "register_buffer"):
        raise ValueError(f"Module {type(module)} does not have register_buffer method")
    module.register_buffer(buffer_name, buffer_value)  # type: ignore[attr-defined]
    logger.debug(f"Registered buffer: {key}")


def _get_transform_settings(entry_name: str, entry_configs: dict[str, Any]) -> list[Any]:
    """Extract transform settings from entry config.

    Args:
        entry_name: Name of the data entry
        entry_configs: Entry configurations

    Returns:
        List of transform settings, empty if none found
    """
    entry_config = entry_configs.get(entry_name)
    if not entry_config:
        return []

    if hasattr(entry_config, "transforms"):
        return entry_config.transforms

    if isinstance(entry_config, dict):
        return entry_config.get("transforms", [])

    return []


def _reconstruct_transform_chain(
    entry_name: str, transform_state: dict[str, Any], entry_configs: dict[str, Any]
) -> TransformChain | None:
    """Reconstruct a TransformChain from state dict.

    Args:
        entry_name: Name of the data entry
        transform_state: State dict for this transform chain
        entry_configs: Entry configurations

    Returns:
        Reconstructed TransformChain or None if reconstruction fails
    """
    try:
        # Get transform settings
        transform_settings = _get_transform_settings(entry_name, entry_configs)
        if not transform_settings:
            logger.warning(f"No transform settings found for '{entry_name}'")
            return None

        # Create and load chain
        chain = TransformChain(transform_settings)
        result = chain.load_state_dict(transform_state, strict=False)

        # Register unexpected keys as buffers (fitted parameters like min/max)
        for key in result.unexpected_keys:
            try:
                _register_transform_buffer(chain, key, transform_state)
            except Exception as e:
                logger.warning(f"Could not register buffer {key}: {e}")

        logger.info(
            f"Loaded transform chain for '{entry_name}' with {len(chain.transforms)} transforms"
        )
        return chain

    except Exception as e:
        logger.error(f"Failed to load transform chain for '{entry_name}': {e}")
        return None
