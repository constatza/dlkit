"""Routines for loading and saving index splits."""

import json
import tomllib
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType

from pydantic import FilePath

from dlkit.infrastructure.types.split import IndexSplit

_SPLIT_READERS: Mapping[str, Callable[[Path], dict]] = MappingProxyType(
    {
        ".json": lambda p: json.loads(p.read_text()),
        ".toml": lambda p: tomllib.loads(p.read_text()),
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
    """Save index splits to a JSON file, adding 'idx_path' metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = idx_split.model_dump(exclude_none=True)
    with path.open("w") as f:
        json.dump(data, f)
