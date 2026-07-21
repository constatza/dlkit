"""Routines for loading and saving index splits."""

import json
import os
import tomllib
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from uuid import uuid4

from pydantic import FilePath

from dlkit.infrastructure.types.split import IndexSplit

_SPLIT_READERS: Mapping[str, Callable[[Path], dict]] = MappingProxyType(
    {
        ".json": lambda p: json.loads(p.read_text()),
        ".toml": lambda p: tomllib.loads(p.read_text()),
    }
)

_SPLIT_WRITERS: Mapping[str, Callable[[IndexSplit], str]] = MappingProxyType(
    {
        ".json": lambda split: json.dumps(split.model_dump(exclude_none=True)),
    }
)


def load_split_indices(path: FilePath) -> IndexSplit:
    """Load train/val/test indices from a JSON or TOML file."""
    suffix = Path(path).suffix.lower()
    reader = _SPLIT_READERS.get(suffix)
    if reader is None:
        raise ValueError(
            f"Unsupported split file format: {suffix!r}. Supported: {sorted(_SPLIT_READERS)}"
        )
    raw = reader(path)
    try:
        return IndexSplit(
            train=raw["train"],
            validation=raw["validation"],
            test=raw["test"],
            predict=raw.get("predict"),
        )
    except KeyError as e:
        raise ValueError(f"Missing key: {e.args[0]} from {path}")


def save_split_indices(
    idx_split: IndexSplit,
    path: Path,
) -> None:
    """Save index splits to a file, atomically. Format is chosen by ``path``'s suffix.

    Writes to a temp file in the same directory and then atomically renames
    it into place via ``os.replace`` — safe when multiple processes (e.g.
    parallel ``optimize()`` trials sharing one split path) resolve the same
    split concurrently. When the destination already exists with
    byte-identical content, this is a silent no-op rather than a redundant
    write, so concurrent writers converge on one value instead of racing.

    Args:
        idx_split: Resolved index split to persist.
        path: Destination file path. Its suffix selects the writer (see
            ``_SPLIT_WRITERS``); only ``.json`` is currently registered.
    """
    suffix = path.suffix.lower()
    writer = _SPLIT_WRITERS.get(suffix)
    if writer is None:
        raise ValueError(
            f"Unsupported split file format: {suffix!r}. Supported: {sorted(_SPLIT_WRITERS)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = writer(idx_split)

    if path.exists() and path.read_text() == payload:
        return

    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex[:8]}.tmp")
    try:
        tmp_path.write_text(payload)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
