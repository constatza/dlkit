"""DLKit TOML source with integrated path preprocessing."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from loguru import logger

from dlkit.infrastructure.config.core._path_helpers import (
    _process_data_paths,
    _process_model_paths,
    _process_training_paths,
    make_path_processor,
)


class DLKitTomlSource:
    """Callable that reads a TOML file and applies DLKit path preprocessing.

    Reads a TOML file, applies unified path resolution (resolving relative paths
    against the config file location and dataset-local roots), and optionally
    filters to a subset of top-level sections.

    Args:
        config_path: Path to the TOML configuration file.
        sections: Optional list of top-level section names to return.
            When ``None`` the full (path-preprocessed) dict is returned.
    """

    def __init__(
        self,
        config_path: Path,
        sections: list[str] | None = None,
    ) -> None:
        self._config_path = Path(config_path)
        self._sections = sections

    def __call__(self) -> dict[str, Any]:
        """Return path-preprocessed TOML data, optionally filtered by section."""
        with open(self._config_path, "rb") as f:
            data = tomllib.load(f)
        data = _preprocess_paths(data, self._config_path)
        if self._sections:
            data = {k: data[k] for k in self._sections if k in data}
        return data


def _preprocess_paths(data: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Unified path resolver for DLKit TOML configs.

    Always processes the full dict before any section filter so that data path
    resolution can reference sibling config sections when needed.

    Handles lowercase JobConfig section names:
        - ``training.trainer.default_root_dir``
        - ``model.checkpoint``
        - ``data.root``
        - ``data.features[*].path`` and ``data.targets[*].path``
        - ``data.splits.filepath``

    Args:
        data: Full TOML dict loaded from disk.
        config_path: Absolute path to the TOML file (used as relative-path base).

    Returns:
        New dict with paths resolved to absolute strings.
    """
    if not data:
        return data

    try:
        from dlkit.infrastructure.types.urls import tilde_expand_strict
    except Exception as exc:
        logger.debug("Path preprocessing skipped (import error): {}", exc)
        return data

    config_dir = config_path.resolve().parent
    root_dir = config_dir if config_dir.exists() else Path.cwd().resolve()
    fn = make_path_processor(root_dir, tilde_expand_strict)

    processed = dict(data)
    processed = _process_training_paths(processed, root_dir, fn)
    processed = _process_model_paths(processed, root_dir, fn)
    processed = _process_data_paths(processed, root_dir, fn, tilde_expand_strict)
    return processed


def _read_env_patches(
    prefix: str,
    delimiter: str = "__",
) -> dict[str, Any]:
    """Read ``os.environ`` into a nested dict for use with :func:`patch_model`.

    Matching is case-insensitive on the prefix. All components after the prefix
    are lowercased to match lowercase section and field names in the JobConfig schema.

    Examples:
        ``DLKIT_RUN__type=train`` with prefix ``"DLKIT"``
        → ``{"run": {"type": "train"}}``

        ``DLKIT_TRAINING__loss=mse`` with prefix ``"DLKIT"``
        → ``{"training": {"loss": "mse"}}``

        ``DLKIT_SESSION__precision=double`` with prefix ``"DLKIT_SESSION__"``
        → ``{"precision": "double"}``

    Args:
        prefix: Env var prefix to strip (case-insensitive match).
        delimiter: Component separator (default ``"__"``).

    Returns:
        Nested dict suitable for :func:`~dlkit.infrastructure.config.core.patching.patch_model`.
    """
    result: dict[str, Any] = {}
    prefix_upper = prefix.upper()

    for key, value in os.environ.items():
        if not key.upper().startswith(prefix_upper):
            continue
        rest = key[len(prefix) :]
        if not rest:
            continue
        if not prefix.endswith(delimiter) and delimiter not in rest:
            continue
        parts = [p for p in rest.split(delimiter) if p]
        if not parts:
            continue

        nested: dict[str, Any] = result
        for i, part in enumerate(parts):
            k = part.lower()
            if i == len(parts) - 1:
                nested[k] = value
            else:
                nested = nested.setdefault(k, {})

    return result
