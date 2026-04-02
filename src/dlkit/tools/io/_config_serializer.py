"""Configuration serialization to TOML format."""

import warnings
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel
from tomlkit import document, dumps, table


def _to_toml_compatible(value: Any) -> Any:
    """Convert values into TOML-serializable primitives.

    - Converts Path to str
    - Converts Enum to its value
    - Converts torch.dtype to str
    - Converts Pydantic models and dataclasses to plain dictionaries
    - Uses ``to_dict()`` when available for runtime value objects such as shape specs
    - Recursively processes dicts and sequences
    """
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return _to_toml_compatible(value.model_dump(exclude_none=True))
    if is_dataclass(value) and not isinstance(value, type):
        return _to_toml_compatible(asdict(value))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            serialized = to_dict()
        except Exception:
            serialized = None
        if serialized is not None:
            return _to_toml_compatible(serialized)
    # Handle pydantic_core Url objects (covers both AnyUrl and Url)
    try:
        from pydantic_core import Url as _CoreUrl

        if isinstance(value, _CoreUrl):
            return str(value)
    except Exception:
        pass
    # Handle torch.dtype objects by converting to string
    if isinstance(value, torch.dtype):
        return str(value)
    # Fallback for torch objects that might not be dtype but still need string conversion
    if hasattr(value, "__module__") and value.__module__ and "torch" in value.__module__:
        return str(value)
    if isinstance(value, dict):
        return {k: _to_toml_compatible(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_to_toml_compatible(v) for v in value if v is not None]
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "__module__") and value.__module__.startswith("dlkit."):
        return str(value)
    return value


def _exclude_value_entries(data: dict[str, Any]) -> dict[str, Any]:
    """Strip in-memory ValueBasedEntry payloads from dataset sections.

    This keeps configs lightweight for logging while preserving structural
    information (names, dtype, transforms, write flags).
    """
    dataset = data.get("DATASET")
    if not isinstance(dataset, dict):
        return data

    sanitized_dataset = dict(dataset)

    def _strip_value_field(entries: Any) -> Any:
        if not isinstance(entries, (list, tuple)):
            return entries
        cleaned: list[Any] = []
        for entry in entries:
            if isinstance(entry, dict):
                cleaned.append({k: v for k, v in entry.items() if k != "value"})
            else:
                cleaned.append(entry)
        return cleaned

    if "features" in sanitized_dataset:
        sanitized_dataset["features"] = _strip_value_field(sanitized_dataset.get("features"))
    if "targets" in sanitized_dataset:
        sanitized_dataset["targets"] = _strip_value_field(sanitized_dataset.get("targets"))

    sanitized = dict(data)
    sanitized["DATASET"] = sanitized_dataset
    return sanitized


def serialize_config_to_string(
    config: BaseModel | dict[str, Any],
    *,
    by_alias: bool = True,
    exclude_none: bool = True,
    exclude_unset: bool = False,
    exclude_value_entries: bool = False,
    sort_sections: bool = True,
) -> str:
    """Serialize a DLKit configuration to a TOML string without writing to disk.

    Accepts a Pydantic model (e.g., GeneralSettings) or a raw dict mapping
    top-level section names to their contents.

    Args:
        config: Pydantic settings model or raw dict to serialize.
        by_alias: Dump using field aliases.
        exclude_none: Exclude fields that are None.
        exclude_unset: Exclude fields that were not explicitly set.
        exclude_value_entries: When True, strip in-memory DataEntry values from Dataset sections.
        sort_sections: Write sections in sorted order for stable diffs.

    Returns:
        TOML-formatted string representation of the config.
    """
    if isinstance(config, BaseModel):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Pydantic serializer warnings:.*",
                category=UserWarning,
            )
            data = config.model_dump(
                by_alias=by_alias, exclude_none=exclude_none, exclude_unset=exclude_unset
            )
    else:
        data = dict(config)

    if exclude_value_entries:
        data = _exclude_value_entries(data)

    doc = document()
    items = sorted(data.items(), key=lambda kv: kv[0]) if sort_sections else data.items()
    for section, content in items:
        if content is None:
            continue
        sec_table = table()
        for k, v in _to_toml_compatible(content).items():
            sec_table.add(k, v)
        doc.add(section, sec_table)

    return dumps(doc)


def write_config(
    config: BaseModel | dict[str, Any],
    output_path: Path | str,
    *,
    by_alias: bool = True,
    exclude_none: bool = True,
    exclude_unset: bool = False,
    exclude_value_entries: bool = False,
    sort_sections: bool = True,
) -> Path:
    """Write a DLKit configuration to a TOML file.

    Accepts a Pydantic model (e.g., GeneralSettings) or a raw dict mapping
    top-level section names (e.g., SESSION, MODEL, PATHS) to their contents.

    Args:
        config: Pydantic settings model or raw dict to write
        output_path: Destination TOML file path
        by_alias: Dump using field aliases (e.g., PATHS.root instead of root_dir)
        exclude_none: Exclude fields that are None
        exclude_unset: Exclude fields that were not explicitly set (for Pydantic models only)
        exclude_value_entries: When True, strip in-memory DataEntry values from Dataset sections
        sort_sections: Write sections in sorted order for stable diffs

    Returns:
        Path to the written TOML file
    """
    output_path = Path(output_path)
    toml_str = serialize_config_to_string(
        config,
        by_alias=by_alias,
        exclude_none=exclude_none,
        exclude_unset=exclude_unset,
        exclude_value_entries=exclude_value_entries,
        sort_sections=sort_sections,
    )
    output_path.write_text(toml_str, encoding="utf-8")
    return output_path
