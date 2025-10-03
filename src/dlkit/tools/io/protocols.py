"""Protocols for config parsing and validation following SOLID principles."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class ConfigParser(Protocol):
    """Protocol for config file parsing strategies.

    Defines the interface for different config parsing approaches,
    enabling strategy pattern implementation for various parsing needs.
    """

    @abstractmethod
    def parse_full(self, config_path: Path | str) -> dict[str, Any]:
        """Parse the entire config file.

        Args:
            config_path: Path to the config file

        Returns:
            Complete config dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is malformed
        """

    @abstractmethod
    def parse_sections(
        self,
        config_path: Path | str,
        section_names: list[str]
    ) -> dict[str, Any]:
        """Parse only specific sections from config file.

        Args:
            config_path: Path to the config file
            section_names: List of section names to extract

        Returns:
            Dictionary containing only requested sections

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is malformed
        """

    @abstractmethod
    def get_available_sections(self, config_path: Path | str) -> list[str]:
        """Get list of available sections without full parsing.

        Args:
            config_path: Path to the config file

        Returns:
            List of section names found in the config

        Raises:
            FileNotFoundError: If config file doesn't exist
        """


@runtime_checkable
class SectionExtractor(Protocol):
    """Protocol for extracting specific sections from parsed config data."""

    @abstractmethod
    def extract_section(
        self,
        config_data: dict[str, Any],
        section_name: str
    ) -> dict[str, Any] | None:
        """Extract a specific section from config data.

        Args:
            config_data: Parsed config dictionary
            section_name: Name of section to extract

        Returns:
            Section data if found, None otherwise
        """

    @abstractmethod
    def extract_sections(
        self,
        config_data: dict[str, Any],
        section_names: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Extract multiple sections from config data.

        Args:
            config_data: Parsed config dictionary
            section_names: List of section names to extract

        Returns:
            Dictionary mapping section names to their data
        """


@runtime_checkable
class ConfigValidator[T: BaseModel](Protocol):
    """Protocol for config validation with Pydantic models."""

    @abstractmethod
    def validate_section(
        self,
        section_data: dict[str, Any],
        model_class: type[T]
    ) -> T:
        """Validate section data with Pydantic model.

        Args:
            section_data: Raw section data
            model_class: Pydantic model class for validation

        Returns:
            Validated model instance

        Raises:
            ValidationError: If validation fails
        """

    @abstractmethod
    def validate_sections(
        self,
        sections_data: dict[str, dict[str, Any]],
        model_classes: dict[str, type[BaseModel]]
    ) -> dict[str, BaseModel]:
        """Validate multiple sections with their corresponding models.

        Args:
            sections_data: Dictionary of section name to section data
            model_classes: Dictionary of section name to model class

        Returns:
            Dictionary of section name to validated model instance

        Raises:
            ValidationError: If any validation fails
        """


@runtime_checkable
class PartialConfigReader(Protocol):
    """High-level protocol for partial config reading operations."""

    @abstractmethod
    def read_section[U: BaseModel](
        self,
        config_path: Path | str,
        model_class: type[U]
    ) -> U:
        """Read and validate a single section.

        Args:
            config_path: Path to config file
            model_class: Pydantic model class for the section

        Returns:
            Validated model instance
        """

    @abstractmethod
    def read_sections(
        self,
        config_path: Path | str,
        section_configs: dict[str, type[BaseModel]]
    ) -> dict[str, BaseModel]:
        """Read and validate multiple sections.

        Args:
            config_path: Path to config file
            section_configs: Mapping of section names to model classes

        Returns:
            Dictionary of section names to validated model instances
        """

    @abstractmethod
    def section_exists(self, config_path: Path | str, section_name: str) -> bool:
        """Check if a section exists without full parsing.

        Args:
            config_path: Path to config file
            section_name: Name of section to check

        Returns:
            True if section exists, False otherwise
        """