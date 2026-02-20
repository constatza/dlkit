"""Transform loading from checkpoints for inference.

Simplified transform loading without unnecessary class wrappers.
Direct functions for loading fitted transforms from checkpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from loguru import logger

from dlkit.core.training.transforms.chain import TransformChain


def load_transforms_from_checkpoint(
    checkpoint: dict[str, Any]
) -> tuple[dict[str, TransformChain], dict[str, TransformChain]]:
    """Load fitted transforms from checkpoint, separated by type.

    Supports two checkpoint formats:
    - Positional (new): ``_feature_chains.0.*`` / ``_target_chains.0.*``
    - String-keyed (legacy): ``fitted_feature_transforms.<name>.*``

    Args:
        checkpoint: Loaded checkpoint dictionary

    Returns:
        Tuple of (feature_transforms, target_transforms) dictionaries
    """
    from dlkit.tools.config.data_entries import is_feature_entry, is_target_entry

    feature_transforms: dict[str, TransformChain] = {}
    target_transforms: dict[str, TransformChain] = {}

    state_dict = checkpoint.get("state_dict", {})
    dlkit_metadata = checkpoint.get("dlkit_metadata", {})
    entry_configs = dlkit_metadata.get("entry_configs", [])

    # Convert list of configs to dict for easier lookup by name
    entry_configs_dict = {
        e["name"]: e for e in entry_configs if isinstance(e, dict) and "name" in e
    }

    # Detect checkpoint format
    has_positional = any(
        k.startswith("_feature_chains.") or k.startswith("_target_chains.")
        for k in state_dict.keys()
    )
    has_modern = any(
        k.startswith("fitted_feature_transforms.") or k.startswith("fitted_target_transforms.")
        for k in state_dict.keys()
    )
    has_legacy = any(k.startswith("fitted_transforms.") for k in state_dict.keys())

    if has_positional:
        logger.info("Loading transforms from positional format (_feature_chains / _target_chains)")
        feature_names: list[str] = dlkit_metadata.get("feature_names", [])
        target_names: list[str] = dlkit_metadata.get("target_names", [])
        feature_transforms = _load_positional_transforms(
            "_feature_chains", state_dict, entry_configs_dict, feature_names
        )
        target_transforms = _load_positional_transforms(
            "_target_chains", state_dict, entry_configs_dict, target_names
        )
    elif has_modern:
        # Guard: No entry configs means we can't load transforms properly
        if not entry_configs_dict:
            logger.warning("No entry_configs found in checkpoint - cannot load transforms")
            return feature_transforms, target_transforms
        logger.info("Loading transforms from modern string-keyed format")
        feature_transforms = _load_transforms_by_prefix(
            "fitted_feature_transforms", state_dict, entry_configs_dict
        )
        target_transforms = _load_transforms_by_prefix(
            "fitted_target_transforms", state_dict, entry_configs_dict
        )
    elif has_legacy:
        if not entry_configs_dict:
            logger.warning("No entry_configs found in checkpoint - cannot load transforms")
            return feature_transforms, target_transforms
        logger.info("Loading transforms from legacy format (will separate)")
        all_transforms = _load_transforms_by_prefix(
            "fitted_transforms", state_dict, entry_configs_dict
        )
        for name, transform in all_transforms.items():
            if name in entry_configs_dict:
                # We need to check entry type from serialized info
                config = entry_configs_dict[name]
                cls_name = config.get("class_name", "")
                if "Feature" in cls_name:
                    feature_transforms[name] = transform
                elif "Target" in cls_name:
                    target_transforms[name] = transform
    else:
        logger.info("No fitted transforms found in checkpoint")

    return feature_transforms, target_transforms


def _load_positional_transforms(
    prefix: str,
    state_dict: dict[str, Any],
    entry_configs: dict[str, Any],
    entry_names: list[str],
) -> dict[str, TransformChain]:
    """Load transforms from positional ModuleList format (_feature_chains / _target_chains).

    Args:
        prefix: State dict prefix (``_feature_chains`` or ``_target_chains``).
        state_dict: Full model state dictionary.
        entry_configs: Entry configurations keyed by name.
        entry_names: Ordered entry names matching positional indices.

    Returns:
        Dictionary mapping entry names to reconstructed TransformChain objects.
    """
    transforms: dict[str, TransformChain] = {}

    # Group state dict entries by positional index
    grouped: dict[int, dict[str, Any]] = {}
    for key, value in state_dict.items():
        if not key.startswith(f"{prefix}."):
            continue
        # key format: "_feature_chains.0.transforms.0._fitted"
        parts = key.split(".", 2)
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        idx = int(parts[1])
        sub_key = parts[2] if len(parts) > 2 else ""
        if idx not in grouped:
            grouped[idx] = {}
        if sub_key:
            grouped[idx][sub_key] = value

    for idx, chain_state in sorted(grouped.items()):
        if idx >= len(entry_names):
            logger.warning(f"Positional index {idx} exceeds entry_names length {len(entry_names)}")
            continue
        entry_name = entry_names[idx]
        chain = _reconstruct_transform_chain(entry_name, chain_state, entry_configs)
        if chain is not None:
            transforms[entry_name] = chain

    return transforms


def _load_transforms_by_prefix(
    prefix: str,
    state_dict: dict[str, Any],
    entry_configs: dict[str, Any]
) -> dict[str, TransformChain]:
    """Load all transforms with given prefix from state_dict.

    Args:
        prefix: Prefix to filter keys (e.g., "fitted_feature_transforms")
        state_dict: Model state dictionary
        entry_configs: Entry configurations

    Returns:
        Dictionary mapping entry names to TransformChain objects
    """
    transforms = {}

    # Group state dict keys by entry name
    transform_state_dicts = _group_by_entry_name(prefix, state_dict)

    # Reconstruct each TransformChain
    for entry_name, transform_state in transform_state_dicts.items():
        chain = _reconstruct_transform_chain(entry_name, transform_state, entry_configs)
        if chain is not None:
            transforms[entry_name] = chain

    return transforms


def _group_by_entry_name(prefix: str, state_dict: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Group state dict keys by entry name.

    Args:
        prefix: Prefix to filter (e.g., "fitted_transforms")
        state_dict: Full state dictionary

    Returns:
        Dictionary mapping entry names to their transform state dicts
    """
    grouped = {}

    for key in state_dict.keys():
        if not key.startswith(f"{prefix}."):
            continue

        # Parse: "prefix.entry_name.param.path" -> "entry_name"
        parts = key.split(".", 2)
        if len(parts) < 2:
            continue

        entry_name = parts[1]
        param_path = parts[2] if len(parts) > 2 else ""

        if entry_name not in grouped:
            grouped[entry_name] = {}

        if param_path:
            grouped[entry_name][param_path] = state_dict[key]

    return grouped


def _register_transform_buffer(chain: TransformChain, key: str, state: dict[str, Any]) -> None:
    """Register a buffer in the transform chain.

    Args:
        chain: The transform chain
        key: Dot-separated key path (e.g., 'transforms.0.min')
        state: State dict containing the buffer value
    """
    from torch.nn import ModuleList

    parts = key.split('.')
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
        elif hasattr(module, 'transforms'):
            # TransformChain has transforms attribute
            module = module.transforms[idx]  # type: ignore[index]
        else:
            raise ValueError(f"Cannot index into {type(module)} at {part}")

    # Register the buffer
    buffer_name = parts[-1]
    buffer_value = state[key]
    if not hasattr(module, 'register_buffer'):
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

    if hasattr(entry_config, 'transforms'):
        return entry_config.transforms

    if isinstance(entry_config, dict):
        return entry_config.get('transforms', [])

    return []


def _reconstruct_transform_chain(
    entry_name: str,
    transform_state: dict[str, Any],
    entry_configs: dict[str, Any]
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

        logger.info(f"Loaded transform chain for '{entry_name}' with {len(chain.transforms)} transforms")
        return chain

    except Exception as e:
        logger.error(f"Failed to load transform chain for '{entry_name}': {e}")
        return None


def apply_transforms(
    data: dict[str, Any],
    transforms: dict[str, TransformChain]
) -> dict[str, Any]:
    """Apply forward transforms to data dictionary.

    Args:
        data: Dictionary of tensors to transform
        transforms: Dictionary of transform chains

    Returns:
        Transformed data dictionary
    """
    transformed = {}
    for name, tensor in data.items():
        if name in transforms:
            transformed[name] = transforms[name](tensor)
        else:
            transformed[name] = tensor
    return transformed


def apply_inverse_transforms(
    data: dict[str, Any] | Any,
    transforms: dict[str, TransformChain]
) -> dict[str, Any] | Any:
    """Apply inverse transforms to data.

    Handles both dict and single tensor cases.
    Detects ambiguity when single tensor with multiple transforms.

    Args:
        data: Data to inverse transform (dict or tensor)
        transforms: Dictionary of transform chains

    Returns:
        Inverse transformed data

    Raises:
        ValueError: If single tensor with multiple transforms (ambiguous)
    """
    # Dict case: apply inverse to each entry
    if isinstance(data, dict):
        return {
            name: transforms[name].inverse_transform(tensor)
            if name in transforms else tensor
            for name, tensor in data.items()
        }

    # Single tensor case: check for ambiguity
    if len(transforms) == 0:
        return data
    elif len(transforms) == 1:
        name = next(iter(transforms.keys()))
        return transforms[name].inverse_transform(data)
    else:
        # Ambiguous: which transform to apply?
        raise ValueError(
            f"Ambiguous inverse transform: model returned single tensor but "
            f"multiple target transforms exist: {list(transforms.keys())}"
        )
