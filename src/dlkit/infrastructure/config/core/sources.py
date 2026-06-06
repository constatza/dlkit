"""DLKit TOML source with integrated path preprocessing."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from loguru import logger


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

    Always processes the full dict before any section filter so that DATASET path
    resolution can reference sibling config sections when needed.

    Handles:
        - ``TRAINING.trainer.default_root_dir``
        - ``MODEL.checkpoint``
        - ``DATASET.split.filepath``
        - ``DATASET.features[*].path`` and ``DATASET.targets[*].path``

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

    def _is_url(value: Any) -> bool:
        return isinstance(value, str) and "://" in value

    def _process_path_field(value: Any, base: Path) -> Any:
        if not isinstance(value, str):
            return value
        if _is_url(value):
            return value
        expanded = tilde_expand_strict(value)
        path_obj = Path(expanded)
        if not path_obj.is_absolute():
            path_obj = (base / path_obj).resolve()
        return str(path_obj)

    processed = dict(data)
    root_dir = config_dir if config_dir.exists() else Path.cwd().resolve()

    # TRAINING.trainer.default_root_dir
    training = processed.get("TRAINING")
    if isinstance(training, dict):
        trainer = training.get("trainer")
        if isinstance(trainer, dict) and "default_root_dir" in trainer:
            trainer_copy = dict(trainer)
            trainer_copy["default_root_dir"] = _process_path_field(
                trainer_copy["default_root_dir"], root_dir
            )
            training_copy = dict(training)
            training_copy["trainer"] = trainer_copy
            processed["TRAINING"] = training_copy

    # MODEL.checkpoint
    model_sec = processed.get("MODEL")
    if isinstance(model_sec, dict) and "checkpoint" in model_sec:
        model_copy = dict(model_sec)
        model_copy["checkpoint"] = _process_path_field(model_copy["checkpoint"], root_dir)
        processed["MODEL"] = model_copy

    # DATASET paths
    dataset_sec = processed.get("DATASET")
    if isinstance(dataset_sec, dict):
        dataset_copy = dict(dataset_sec)
        dataset_root_val = dataset_copy.get("root_dir", dataset_copy.get("root"))
        dataset_base: Path | None = None
        if isinstance(dataset_root_val, str) and dataset_root_val:
            processed_root = _process_path_field(dataset_root_val, root_dir)
            dataset_copy["root_dir"] = processed_root
            dataset_base = Path(processed_root)

        split = dataset_copy.get("split")
        if isinstance(split, dict) and "filepath" in split:
            split_copy = dict(split)
            value = split_copy["filepath"]
            if isinstance(value, str) and dataset_base and not _is_url(value):
                p = Path(tilde_expand_strict(value))
                if not p.is_absolute():
                    split_copy["filepath"] = str((dataset_base / p).resolve())
                else:
                    split_copy["filepath"] = str(p)
            else:
                split_copy["filepath"] = _process_path_field(value, root_dir)
            dataset_copy["split"] = split_copy

        for list_key in ("features", "targets"):
            entries = dataset_copy.get(list_key)
            if isinstance(entries, (list, tuple)):
                new_entries = []
                changed = False
                for item in entries:
                    if not isinstance(item, dict):
                        new_entries.append(item)
                        continue
                    if "path" in item:
                        new_item = dict(item)
                        path_val = new_item["path"]
                        if isinstance(path_val, str) and dataset_base and not _is_url(path_val):
                            p_val = Path(tilde_expand_strict(path_val))
                            if not p_val.is_absolute():
                                new_item["path"] = str((dataset_base / p_val).resolve())
                            else:
                                new_item["path"] = str(p_val)
                        else:
                            new_item["path"] = _process_path_field(path_val, root_dir)
                        if new_item["path"] != item.get("path"):
                            changed = True
                        new_entries.append(new_item)
                    else:
                        new_entries.append(item)
                if changed:
                    dataset_copy[list_key] = new_entries

        processed["DATASET"] = dataset_copy

    return processed


def _read_env_patches(prefix: str, delimiter: str = "__") -> dict[str, Any]:
    """Read ``os.environ`` into a nested dict for use with :func:`patch_model`.

    Matching is case-insensitive on the prefix. The first component after the
    prefix is uppercased (section name); all remaining components are lowercased
    (field names). When only one component remains after stripping the prefix
    (i.e. the prefix already encoded the section name), that component is
    lowercased so it matches Pydantic field names.

    Examples:
        ``DLKIT_SESSION__precision=double`` with prefix ``"DLKIT_"``
        → ``{"SESSION": {"precision": "double"}}``

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
        # Skip flat vars unless the prefix already consumed the section
        # component (i.e. ends with the delimiter).
        if not prefix.endswith(delimiter) and delimiter not in rest:
            continue
        parts = [p for p in rest.split(delimiter) if p]
        if not parts:
            continue

        nested: dict[str, Any] = result
        for i, part in enumerate(parts):
            # Multiple components: first is a section name (uppercase),
            # rest are field names (lowercase).
            # Single component: it's a field name (lowercase).
            k = part.upper() if (i == 0 and len(parts) > 1) else part.lower()
            if i == len(parts) - 1:
                nested[k] = value
            else:
                nested = nested.setdefault(k, {})

    return result
