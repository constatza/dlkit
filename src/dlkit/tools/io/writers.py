"""General-purpose file writers for TOML, JSON, and YAML formats.

Provides a protocol-based writer system with a factory for creating format-specific
writers. Writers handle serialization only — pre-processing (Pydantic model_dump,
exclusions, sorting) is the caller's responsibility.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


def _normalize_to_dict(data: Any) -> dict:
    """Convert common data containers to a plain dict.

    Args:
        data: A Pydantic BaseModel, dataclass, Mapping, or plain dict.

    Returns:
        Plain dict representation of the data.

    Raises:
        TypeError: If data cannot be converted to a dict.
    """
    if isinstance(data, BaseModel):
        return data.model_dump()
    if is_dataclass(data) and not isinstance(data, type):
        return asdict(data)
    if isinstance(data, Mapping):
        return dict(data)
    raise TypeError(f"Cannot normalize {type(data).__name__!r} to dict")


@runtime_checkable
class IWriter(Protocol):
    """Protocol for format-specific file writers.

    Writers accept already-processed dicts. They do not perform
    Pydantic model_dump, field exclusions, or sorting.
    """

    extension: str

    def to_string(self, data: Any) -> str:
        """Serialize data to a string.

        Args:
            data: Data to serialize (dict or dict-compatible object).

        Returns:
            String representation in the writer's format.
        """
        ...

    def write(self, data: Any, path: Path) -> Path:
        """Serialize data and write to a file.

        Args:
            data: Data to serialize.
            path: Destination file path.

        Returns:
            The path that was written.
        """
        ...


class TomlWriter:
    """Writes data in TOML format using tomlkit.

    Uses tomlkit for pretty-printed, comment-preserving TOML output.
    Accepts plain dicts (not Pydantic models — use _normalize_to_dict first).
    """

    extension: str = "toml"

    def to_string(self, data: Any) -> str:
        """Serialize data to a TOML string.

        Args:
            data: Plain dict mapping section names to their contents.

        Returns:
            TOML-formatted string.
        """
        from tomlkit import dumps

        d = _normalize_to_dict(data) if not isinstance(data, dict) else data
        return dumps(d)

    def write(self, data: Any, path: Path) -> Path:
        """Serialize data and write to a TOML file.

        Args:
            data: Data to serialize.
            path: Destination file path.

        Returns:
            The path that was written.
        """
        path = Path(path)
        path.write_text(self.to_string(data), encoding="utf-8")
        return path


class JsonWriter:
    """Writes data in JSON format using the standard library.

    Produces indented, human-readable JSON output.
    """

    extension: str = "json"

    def to_string(self, data: Any) -> str:
        """Serialize data to a JSON string.

        Args:
            data: Data to serialize (must be JSON-serializable).

        Returns:
            JSON-formatted string with 2-space indentation.
        """
        d = _normalize_to_dict(data) if not isinstance(data, dict) else data
        return json.dumps(d, indent=2, default=str)

    def write(self, data: Any, path: Path) -> Path:
        """Serialize data and write to a JSON file.

        Args:
            data: Data to serialize.
            path: Destination file path.

        Returns:
            The path that was written.
        """
        path = Path(path)
        path.write_text(self.to_string(data), encoding="utf-8")
        return path


class YamlWriter:
    """Writes data in YAML format using PyYAML.

    PyYAML is an optional dependency; ImportError is raised at call time
    if it is not installed.
    """

    extension: str = "yaml"

    def to_string(self, data: Any) -> str:
        """Serialize data to a YAML string.

        Args:
            data: Data to serialize.

        Returns:
            YAML-formatted string.

        Raises:
            ImportError: If PyYAML is not installed.
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for YamlWriter. Install it with: pip install pyyaml"
            ) from exc
        d = _normalize_to_dict(data) if not isinstance(data, dict) else data
        return yaml.dump(d, default_flow_style=False, allow_unicode=True)

    def write(self, data: Any, path: Path) -> Path:
        """Serialize data and write to a YAML file.

        Args:
            data: Data to serialize.
            path: Destination file path.

        Returns:
            The path that was written.

        Raises:
            ImportError: If PyYAML is not installed.
        """
        path = Path(path)
        path.write_text(self.to_string(data), encoding="utf-8")
        return path


class WriterFactory:
    """Factory for creating format-specific writers.

    Supports "toml", "json", and "yaml" formats.
    """

    @staticmethod
    def create(format: str = "toml") -> IWriter:  # noqa: A002
        """Create a writer for the given format.

        Args:
            format: Output format — one of "toml", "json", "yaml".

        Returns:
            Writer instance for the requested format.

        Raises:
            ValueError: If the format is not supported.
        """
        match format.lower():
            case "toml":
                return TomlWriter()
            case "json":
                return JsonWriter()
            case "yaml" | "yml":
                return YamlWriter()
            case _:
                raise ValueError(
                    f"Unsupported format {format!r}. Choose from: toml, json, yaml"
                )
