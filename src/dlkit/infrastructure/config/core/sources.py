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

    Always processes the full dict before any section filter so that data path
    resolution can reference sibling config sections when needed.

    Handles new lowercase section names (JobConfig schema):
        - ``training.trainer.default_root_dir``
        - ``model.checkpoint``
        - ``data.root``
        - ``data.features[*].path`` and ``data.targets[*].path``
        - ``data.splits.filepath``

    Also handles legacy uppercase section names for backward compatibility:
        - ``TRAINING.trainer.default_root_dir``
        - ``MODEL.checkpoint``
        - ``DATASET.root_dir`` / ``DATASET.root``
        - ``DATASET.features[*].path`` and ``DATASET.targets[*].path``
        - ``DATAMODULE.split.filepath``

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

    # training.trainer.default_root_dir  (new lowercase schema)
    training = processed.get("training")
    if isinstance(training, dict):
        trainer = training.get("trainer")
        if isinstance(trainer, dict) and "default_root_dir" in trainer:
            trainer_copy = dict(trainer)
            trainer_copy["default_root_dir"] = _process_path_field(
                trainer_copy["default_root_dir"], root_dir
            )
            training_copy = dict(training)
            training_copy["trainer"] = trainer_copy
            processed["training"] = training_copy

    # TRAINING.trainer.default_root_dir  (legacy uppercase schema)
    training_upper = processed.get("TRAINING")
    if isinstance(training_upper, dict):
        trainer = training_upper.get("trainer")
        if isinstance(trainer, dict) and "default_root_dir" in trainer:
            trainer_copy = dict(trainer)
            trainer_copy["default_root_dir"] = _process_path_field(
                trainer_copy["default_root_dir"], root_dir
            )
            training_copy = dict(training_upper)
            training_copy["trainer"] = trainer_copy
            processed["TRAINING"] = training_copy

    # model.checkpoint  (new lowercase schema)
    model_sec = processed.get("model")
    if isinstance(model_sec, dict) and "checkpoint" in model_sec:
        model_copy = dict(model_sec)
        model_copy["checkpoint"] = _process_path_field(model_copy["checkpoint"], root_dir)
        processed["model"] = model_copy

    # MODEL.checkpoint  (legacy uppercase schema)
    model_sec_upper = processed.get("MODEL")
    if isinstance(model_sec_upper, dict) and "checkpoint" in model_sec_upper:
        model_copy = dict(model_sec_upper)
        model_copy["checkpoint"] = _process_path_field(model_copy["checkpoint"], root_dir)
        processed["MODEL"] = model_copy

    # data paths  (new lowercase schema: data.root, data.features/targets, data.splits.filepath)
    data_sec = processed.get("data")
    if isinstance(data_sec, dict):
        data_copy = dict(data_sec)
        data_root_val = data_copy.get("root")
        data_base: Path | None = None
        if isinstance(data_root_val, str) and data_root_val:
            processed_root = _process_path_field(data_root_val, root_dir)
            data_copy["root"] = processed_root
            data_base = Path(processed_root)

        for list_key in ("features", "targets"):
            entries = data_copy.get(list_key)
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
                        if isinstance(path_val, str) and data_base and not _is_url(path_val):
                            p_val = Path(tilde_expand_strict(path_val))
                            if not p_val.is_absolute():
                                new_item["path"] = str((data_base / p_val).resolve())
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
                    data_copy[list_key] = new_entries

        # data.splits.filepath
        splits = data_copy.get("splits")
        if isinstance(splits, dict) and "filepath" in splits:
            splits_copy = dict(splits)
            splits_copy["filepath"] = _process_path_field(splits_copy["filepath"], root_dir)
            data_copy["splits"] = splits_copy

        processed["data"] = data_copy

    # DATASET paths  (legacy uppercase schema)
    dataset_sec = processed.get("DATASET")
    if isinstance(dataset_sec, dict):
        dataset_copy = dict(dataset_sec)
        dataset_root_val = dataset_copy.get("root_dir", dataset_copy.get("root"))
        dataset_base: Path | None = None
        if isinstance(dataset_root_val, str) and dataset_root_val:
            processed_root = _process_path_field(dataset_root_val, root_dir)
            dataset_copy["root_dir"] = processed_root
            dataset_base = Path(processed_root)

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

    # DATAMODULE.split.filepath  (legacy uppercase schema)
    datamodule_sec = processed.get("DATAMODULE")
    if isinstance(datamodule_sec, dict):
        datamodule_copy = dict(datamodule_sec)
        split = datamodule_copy.get("split")
        if isinstance(split, dict) and "filepath" in split:
            split_copy = dict(split)
            value = split_copy["filepath"]
            split_copy["filepath"] = _process_path_field(value, root_dir)
            datamodule_copy["split"] = split_copy
        processed["DATAMODULE"] = datamodule_copy

    return processed


def _read_env_patches(prefix: str, delimiter: str = "__") -> dict[str, Any]:
    """Read ``os.environ`` into a nested dict for use with :func:`patch_model`.

    Matching is case-insensitive on the prefix. All components after the prefix
    are lowercased so they match lowercase section and field names in the new
    JobConfig schema.

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
        # Skip flat vars unless the prefix already consumed the section
        # component (i.e. ends with the delimiter).
        if not prefix.endswith(delimiter) and delimiter not in rest:
            continue
        parts = [p for p in rest.split(delimiter) if p]
        if not parts:
            continue

        nested: dict[str, Any] = result
        for i, part in enumerate(parts):
            # All components are lowercased to match the new lowercase schema.
            k = part.lower()
            if i == len(parts) - 1:
                nested[k] = value
            else:
                nested = nested.setdefault(k, {})

    return result
