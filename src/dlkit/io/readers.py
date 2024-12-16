from typing import Dict, Any
import numpy as np
import torch
from pydantic import validate_call, FilePath
from pathlib import Path, PurePath
import re
import tomllib


def check_paths(paths_dict: dict) -> dict:
    new_dict = {}
    for key, path_name in paths_dict.items():
        if path_name.startswith("sqlite") or path_name.startswith("postgresql"):
            path = str(path_name)
        else:
            path = Path(path_name)
        new_dict[key] = path
    return new_dict


@validate_call
def read_toml(config_path: FilePath) -> Dict:
    """Reads a TOML configuration file."""
    with open(config_path, "r") as file:
        content = file.read()
    parsed = parse_self_referencing_toml(content)
    return parsed


def parse_self_referencing_toml(toml_string: str) -> Dict:
    """
    Parses a TOML string and resolves self-references in the format {table.key}.

    Args:
        toml_string (str): The TOML input string with potential self-references.

    Returns:
        dict: The resolved TOML data as a dictionary.
    """
    data = tomllib.loads(toml_string)
    resolving = set()  # To track and prevent circular references
    pattern = re.compile(r"\{(.+?)\}")  # Regex pattern to find {table.key}

    def resolve(value, full_data):
        """Recursively resolve a single value (either string or other data types)."""
        if isinstance(value, str):
            matches = pattern.findall(value)
            for match in matches:
                # Resolve hierarchical references, e.g., "table.subtable.key"
                keys = match.split(".")
                current = full_data

                try:
                    # Traverse the nested dictionary
                    for key in keys:
                        current = current[key]
                except KeyError:
                    raise KeyError(f"Reference '{match}' not found in the TOML data.")

                if isinstance(current, str) and pattern.search(current):
                    # Resolve recursively if the referenced value itself has placeholders
                    if tuple(keys) in resolving:
                        raise ValueError(f"Circular reference detected in '{match}'.")
                    resolving.add(tuple(keys))
                    resolved_value = resolve(current, full_data)
                    resolving.remove(tuple(keys))
                else:
                    resolved_value = current

                # Replace placeholder with the actual resolved value
                value = value.replace(f"{{{match}}}", str(resolved_value))
        return value

    def resolve_all():
        """Recursively resolve all references in the TOML data."""

        def traverse_and_resolve(data, full_data):
            if isinstance(data, dict):
                return {
                    key: traverse_and_resolve(value, full_data)
                    for key, value in data.items()
                }
            elif isinstance(data, list):
                return [traverse_and_resolve(item, full_data) for item in data]
            else:
                return resolve(data, full_data)

        return traverse_and_resolve(data, data)

    # Resolve all self-references in the TOML structure
    resolved_data = resolve_all()
    return resolved_data


@validate_call
def load_config(config_path: FilePath) -> dict:
    config = read_toml(config_path)
    config = dict(config)
    config["paths"] = check_paths(config["paths"])
    return config


@validate_call
def read_array_as_numpy(path: FilePath):
    if path.suffix == ".npy":
        return np.load(path)
    elif path.suffix == ".csv":
        return np.loadtxt(path, delimiter=",")
    elif path.suffix == ".pt":
        return torch.load(path).numpy()
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
