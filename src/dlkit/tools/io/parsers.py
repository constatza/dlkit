"""Config parsing implementations for efficient partial reading."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from dynaconf import Dynaconf


class PartialTOMLParser:
    """Efficient TOML parser that can read only specific sections.

    This parser scans the file to find section boundaries and extracts
    only the requested sections, avoiding full file parsing when possible.
    """

    def __init__(self):
        self._section_pattern = re.compile(r'^\s*\[([^\]]+)\]')
        self._nested_section_pattern = re.compile(r'^\s*\[\[([^\]]+)\]\]')

    def parse_full(self, config_path: Path | str) -> dict[str, Any]:
        """Parse the entire config file using dynaconf."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        settings = Dynaconf(
            settings_files=[str(config_path)],
            load_dotenv=False,
        )

        config_data = settings.to_dict()
        return self._clean_dynaconf_metadata(config_data)

    def parse_sections(
        self,
        config_path: Path | str,
        section_names: list[str]
    ) -> dict[str, Any]:
        """Parse only specific sections from config file.

        Uses efficient line-by-line scanning to extract only requested sections.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # For small number of sections, use targeted extraction
        if len(section_names) <= 3:
            return self._extract_sections_targeted(config_path, section_names)

        # For many sections, fall back to full parsing
        full_config = self.parse_full(config_path)
        return {name: full_config.get(name, {}) for name in section_names}

    def get_available_sections(self, config_path: Path | str) -> list[str]:
        """Get list of available sections without full parsing."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        sections = []
        with open(config_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Check for section headers
                match = self._section_pattern.match(line)
                if match:
                    section_name = match.group(1)
                    sections.append(section_name)

                # Check for array of tables
                match = self._nested_section_pattern.match(line)
                if match:
                    section_name = match.group(1)
                    if section_name not in sections:
                        sections.append(section_name)

        return sections

    def _extract_sections_targeted(
        self,
        config_path: Path,
        section_names: list[str]
    ) -> dict[str, Any]:
        """Extract specific sections using targeted line-by-line parsing.

        Handles nested sections like [MLFLOW.server] as part of the [MLFLOW] section.
        """
        # For sections with potential nested subsections, fall back to full parsing
        # to ensure proper handling of nested TOML structures like [MLFLOW.server]
        has_potentially_nested_sections = any(
            section_name in {'MLFLOW', 'OPTUNA', 'TRAINING', 'MODEL', 'DATASET'}
            for section_name in section_names
        )

        if has_potentially_nested_sections:
            # Use full parsing to properly handle nested sections
            full_config = self.parse_full(config_path)
            return {name: full_config.get(name, {}) for name in section_names}

        # For simple sections without nesting, use the original targeted approach
        sections_data = {}
        current_section = None
        current_content = []

        # Normalize section names to handle nested sections
        target_sections = set(section_names)

        with open(config_path, encoding='utf-8') as f:
            for line in f:
                line.strip()

                # Check for section headers
                section_match = self._section_pattern.match(line)
                nested_match = self._nested_section_pattern.match(line)

                if section_match or nested_match:
                    # Save previous section if it was a target
                    if current_section and current_section in target_sections:
                        section_toml = '\n'.join([f'[{current_section}]'] + current_content)
                        sections_data[current_section] = self._parse_toml_section(section_toml)

                    # Start new section
                    if section_match:
                        current_section = section_match.group(1)
                    else:
                        current_section = nested_match.group(1)
                    current_content = []

                elif current_section and current_section in target_sections:
                    # Collect content for target sections
                    current_content.append(line.rstrip())

        # Handle last section
        if current_section and current_section in target_sections:
            section_toml = '\n'.join([f'[{current_section}]'] + current_content)
            sections_data[current_section] = self._parse_toml_section(section_toml)

        return sections_data

    def _parse_toml_section(self, section_toml: str) -> dict[str, Any]:
        """Parse a single section's TOML content."""
        try:
            # Create a temporary file-like object for dynaconf
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
                f.write(section_toml)
                temp_path = f.name

            try:
                settings = Dynaconf(
                    settings_files=[temp_path],
                    load_dotenv=False,
                )
                config_data = settings.to_dict()

                # Extract the section content (remove the section wrapper)
                for section_name, section_data in config_data.items():
                    if section_name not in {'LOAD_DOTENV', 'ENV_FOR_DYNACONF', 'ROOT_PATH_FOR_DYNACONF'}:
                        return section_data
                return {}
            finally:
                os.unlink(temp_path)

        except Exception:
            # Fallback to empty dict if parsing fails
            return {}

    def _clean_dynaconf_metadata(self, config_data: dict[str, Any]) -> dict[str, Any]:
        """Remove dynaconf metadata from config data."""
        dynaconf_metadata_keys = {
            "LOAD_DOTENV",
            "ENV_FOR_DYNACONF",
            "ROOT_PATH_FOR_DYNACONF"
        }
        return {k: v for k, v in config_data.items() if k not in dynaconf_metadata_keys}


class DynafconfConfigParser:
    """Full dynaconf-based parser for compatibility."""

    def parse_full(self, config_path: Path | str) -> dict[str, Any]:
        """Parse entire config using dynaconf."""
        parser = PartialTOMLParser()
        return parser.parse_full(config_path)

    def parse_sections(
        self,
        config_path: Path | str,
        section_names: list[str]
    ) -> dict[str, Any]:
        """Parse all sections then filter (less efficient but compatible)."""
        full_config = self.parse_full(config_path)
        return {name: full_config.get(name, {}) for name in section_names}

    def get_available_sections(self, config_path: Path | str) -> list[str]:
        """Get sections by parsing full config."""
        full_config = self.parse_full(config_path)
        return list(full_config.keys())


class StandardSectionExtractor:
    """Standard implementation for extracting sections from parsed config."""

    def extract_section(
        self,
        config_data: dict[str, Any],
        section_name: str
    ) -> dict[str, Any] | None:
        """Extract a single section from config data."""
        return config_data.get(section_name)

    def extract_sections(
        self,
        config_data: dict[str, Any],
        section_names: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Extract multiple sections from config data."""
        result = {}
        for name in section_names:
            section_data = self.extract_section(config_data, name)
            if section_data is not None:
                result[name] = section_data
        return result