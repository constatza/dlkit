#!/usr/bin/env python3
"""
Demonstration of the new partial config reading functionality.

This script shows how to use the efficient partial config reader to load
only specific sections from TOML config files without parsing the entire file.
"""

from pathlib import Path
from dlkit.tools.io.config import (
    load_section_config,
    load_sections_config,
    check_section_exists,
    get_available_sections,
)
from dlkit.tools.config.paths_settings import PathsSettings
from dlkit.tools.config.components.model_components import ModelComponentSettings

def main():
    """Demonstrate partial config reading capabilities."""
    config_path = Path("config/config.toml")

    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        print("Please run this script from the project root directory.")
        return

    print("=== Partial Config Reader Demo ===\n")

    # 1. Check available sections without parsing the entire file
    print("1. Available sections in config file:")
    sections = get_available_sections(config_path)
    for section in sections:
        print(f"   - {section}")
    print()

    # 2. Check if specific sections exist
    print("2. Checking if specific sections exist:")
    sections_to_check = ["PATHS", "MODEL", "NONEXISTENT"]
    for section in sections_to_check:
        exists = check_section_exists(config_path, section)
        print(f"   {section}: {'✓' if exists else '✗'}")
    print()

    # 3. Load a single section efficiently
    print("3. Loading single section (PATHS) without specifying the model class:")
    if check_section_exists(config_path, "PATHS"):
        try:
            paths_config = load_section_config(config_path, section_name="PATHS")
            print(f"   Loaded {type(paths_config).__name__}:")
            print(f"   - dataroot: {paths_config.dataroot}")
            print(f"   - input: {paths_config.input}")
            print(f"   - output: {paths_config.output}")
            print(f"   - typed PathsSettings: {isinstance(paths_config, PathsSettings)}")
        except Exception as e:
            print(f"   Error: {e}")
    print()

    # 4. Load multiple sections efficiently
    print("4. Loading multiple sections with the registry shortcut:")
    try:
        configs = load_sections_config(config_path, ["PATHS", "MODEL"])

        print(f"   Loaded {len(configs)} sections:")
        for section_name, config in configs.items():
            print(f"   - {section_name}: {type(config).__name__}")
            if isinstance(config, ModelComponentSettings):
                print(f"     • model name: {config.name}")
    except Exception as e:
        print(f"   Error: {e}")
    print()

    # 5. Performance comparison example
    print("5. Performance benefits:")
    print("   ✓ Partial loading avoids parsing unused sections")
    print("   ✓ Faster startup for applications that only need specific config")
    print("   ✓ Lower memory usage")
    print("   ✓ Better error isolation - only relevant sections need to be valid")
    print()

    print("=== Demo Complete ===")

if __name__ == "__main__":
    main()
