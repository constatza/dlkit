"""Routines for loading and saving index splits."""

from pydantic import FilePath
from pathlib import Path
import json
from dlkit.core.datatypes.split import IndexSplit


def load_split_indices(path: FilePath) -> IndexSplit:
    """Load train/val/test indices from a JSON file."""
    with path.open("r") as f:
        raw = json.load(f)
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
