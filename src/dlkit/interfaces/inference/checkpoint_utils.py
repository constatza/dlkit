"""Shared utilities for checkpoint loading and state dict extraction.

Provides pure functions for checkpoint processing following functional
programming principles with SOLID design.
"""

from __future__ import annotations

from typing import Dict, Any
from loguru import logger


# Pure functions (no side effects except logging)

def _has_model_prefix(keys: list[str]) -> bool:
    """Check if any key has 'model.' prefix (pure function)."""
    return any(k.startswith("model.") for k in keys)


def _strip_prefix_if_present(key: str, prefix: str = "model.") -> str:
    """Strip prefix from key if present (pure function)."""
    return key.replace(prefix, "", 1) if key.startswith(prefix) else key


def _strip_model_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Strip 'model.' prefix from keys that have it (pure function)."""
    return {_strip_prefix_if_present(k): v for k, v in state_dict.items()}


def _extract_raw_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Extract raw state dict from checkpoint (pure function)."""
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def extract_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Extract state dict with automatic prefix stripping.

    Composition of pure functions. Only side effect: logging.

    Args:
        checkpoint: Checkpoint dictionary from torch.load()

    Returns:
        State dict with "model." prefix stripped from model weight keys
    """
    state_dict = _extract_raw_state_dict(checkpoint)

    if not isinstance(state_dict, dict):
        return state_dict

    if _has_model_prefix(list(state_dict.keys())):
        logger.info("Stripping 'model.' prefix from state dict keys")
        return _strip_model_prefix(state_dict)

    return state_dict
