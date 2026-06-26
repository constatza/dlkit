"""Pure path-processing helpers for DLKit config source loading."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any


def make_path_processor(
    root_dir: Path,
    expand_fn: Callable[[str], str],
) -> Callable[[Any, Path], Any]:
    """Return a function that expands and rebases a path against root_dir.

    Args:
        root_dir: Base directory to resolve relative paths against.
        expand_fn: Function that expands tilde and environment variables in a
            path string.

    Returns:
        A callable ``(value, base) -> Any`` that leaves non-strings and URLs
        unchanged and resolves relative path strings to absolute strings.
    """

    def _is_url(value: Any) -> bool:
        return isinstance(value, str) and "://" in value

    def process(value: Any, base: Path) -> Any:
        """Expand and rebase a single path value.

        Args:
            value: Raw config value (may not be a string path).
            base: Directory to resolve relative paths against.

        Returns:
            Absolute path string when *value* is a relative path string,
            otherwise the original value unchanged.
        """
        if not isinstance(value, str):
            return value
        if _is_url(value):
            return value
        expanded = expand_fn(value)
        path_obj = Path(expanded)
        if not path_obj.is_absolute():
            path_obj = (base / path_obj).resolve()
        return str(path_obj)

    return process


def _process_entry_paths(
    entries: list[Any] | tuple[Any, ...],
    data_base: Path | None,
    root_dir: Path,
    fn: Callable[[Any, Path], Any],
    expand_fn: Callable[[str], str],
) -> tuple[list[Any], bool]:
    """Rebase path fields inside a list of entry dicts.

    Args:
        entries: Sequence of feature/target entry dicts.
        data_base: Resolved data root directory (used as base when present).
        root_dir: Config-file parent directory (fallback base).
        fn: Path-processor callable from :func:`make_path_processor`.
        expand_fn: Tilde-expand callable.

    Returns:
        Tuple of ``(new_entries, changed)`` where *changed* is ``True`` when
        at least one entry was modified.
    """

    def _is_url(value: Any) -> bool:
        return isinstance(value, str) and "://" in value

    new_entries: list[Any] = []
    changed = False
    for item in entries:
        if not isinstance(item, dict):
            new_entries.append(item)
            continue
        if "path" not in item:
            new_entries.append(item)
            continue
        new_item = dict(item)
        path_val = new_item["path"]
        if isinstance(path_val, str) and data_base and not _is_url(path_val):
            p_val = Path(expand_fn(path_val))
            if not p_val.is_absolute():
                new_item["path"] = str((data_base / p_val).resolve())
            else:
                new_item["path"] = str(p_val)
        else:
            new_item["path"] = fn(path_val, root_dir)
        if new_item["path"] != item.get("path"):
            changed = True
        new_entries.append(new_item)
    return new_entries, changed


def _process_training_paths(
    processed: dict[str, Any],
    root_dir: Path,
    fn: Callable[[Any, Path], Any],
) -> dict[str, Any]:
    """Rebase training.trainer.default_root_dir against the config root.

    Args:
        processed: Mutable copy of the top-level config dict.
        root_dir: Directory to resolve relative paths against.
        fn: Path-processor callable from :func:`make_path_processor`.

    Returns:
        Updated config dict (new dict, input is not mutated).
    """
    training = processed.get("training")
    if not isinstance(training, dict):
        return processed
    trainer = training.get("trainer")
    if not isinstance(trainer, dict) or "default_root_dir" not in trainer:
        return processed
    trainer_copy = dict(trainer)
    trainer_copy["default_root_dir"] = fn(trainer_copy["default_root_dir"], root_dir)
    training_copy = dict(training)
    training_copy["trainer"] = trainer_copy
    return {**processed, "training": training_copy}


def _process_model_paths(
    processed: dict[str, Any],
    root_dir: Path,
    fn: Callable[[Any, Path], Any],
) -> dict[str, Any]:
    """Rebase model.checkpoint against the config root.

    Args:
        processed: Mutable copy of the top-level config dict.
        root_dir: Directory to resolve relative paths against.
        fn: Path-processor callable from :func:`make_path_processor`.

    Returns:
        Updated config dict (new dict, input is not mutated).
    """
    model_sec = processed.get("model")
    if not isinstance(model_sec, dict) or "checkpoint" not in model_sec:
        return processed
    model_copy = dict(model_sec)
    model_copy["checkpoint"] = fn(model_copy["checkpoint"], root_dir)
    return {**processed, "model": model_copy}


def _process_data_paths(
    processed: dict[str, Any],
    root_dir: Path,
    fn: Callable[[Any, Path], Any],
    expand_fn: Callable[[str], str],
) -> dict[str, Any]:
    """Rebase data.root, data.features[*].path, data.targets[*].path, data.splits.filepath.

    Args:
        processed: Mutable copy of the top-level config dict.
        root_dir: Directory to resolve relative paths against.
        fn: Path-processor callable from :func:`make_path_processor`.
        expand_fn: Tilde-expand callable (needed for entry-level path rebasing).

    Returns:
        Updated config dict (new dict, input is not mutated).
    """
    data_sec = processed.get("data")
    if not isinstance(data_sec, dict):
        return processed

    data_copy = dict(data_sec)
    data_base: Path | None = None

    data_root_val = data_copy.get("root")
    if isinstance(data_root_val, str) and data_root_val:
        processed_root = fn(data_root_val, root_dir)
        data_copy["root"] = processed_root
        data_base = Path(processed_root)

    for list_key in ("features", "targets"):
        entries = data_copy.get(list_key)
        if not isinstance(entries, (list, tuple)):
            continue
        new_entries, changed = _process_entry_paths(entries, data_base, root_dir, fn, expand_fn)
        if changed:
            data_copy[list_key] = new_entries

    splits = data_copy.get("splits")
    if isinstance(splits, dict) and "filepath" in splits:
        splits_copy = dict(splits)
        splits_copy["filepath"] = fn(splits_copy["filepath"], root_dir)
        data_copy["splits"] = splits_copy

    return {**processed, "data": data_copy}
