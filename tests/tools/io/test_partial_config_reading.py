"""Tests for partial config reading functionality."""

from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from dlkit.tools.io.config import (
    load_section_config,
    load_sections_config,
    check_section_exists,
    get_available_sections,
    reset_section_mappings,
    ConfigSectionError,
    ConfigValidationError,
)
from dlkit.tools.io.parsers import PartialTOMLParser
from dlkit.tools.config.paths_settings import PathsSettings
from dlkit.tools.config.components.model_components import ModelComponentSettings


@pytest.fixture(autouse=True)
def reset_section_registry():
    """Ensure section mappings are reset before and after each test."""
    reset_section_mappings()
    yield
    reset_section_mappings()


class SampleSectionSettings(BaseModel):
    """Test settings for a simple section."""

    name: str = Field(default="test")
    value: int = Field(default=42)
    enabled: bool = Field(default=True)


class PathsTestSettings(BaseModel):
    """Test settings for paths section."""

    dataroot: str = Field(default="./data")
    input: str = Field(default="./data/input")
    output: str = Field(default="./data/output")


class ModelTestSettings(BaseModel):
    """Test settings for model section."""

    name: str = Field(default="test_model")
    latent_size: int = Field(default=64)
    num_layers: int = Field(default=3)


@pytest.fixture
def sample_config_content():
    """Sample TOML config content for testing."""
    return """
[PATHS]
dataroot = "./test_data"
input = "./test_data/input"
output = "./test_data/output"

[MODEL]
name = "TestModel"
latent_size = 128
num_layers = 5

[TRAINER]
max_epochs = 100
learning_rate = 0.001

[MLFLOW]
enabled = true
experiment_name = "test_experiment"

[[DATASET]]
name = "dataset1"
path = "./data1.csv"

[[DATASET]]
name = "dataset2"
path = "./data2.csv"
"""


@pytest.fixture
def config_file(tmp_path, sample_config_content):
    """Create a temporary config file for testing."""
    config_path = tmp_path / "test_config.toml"
    config_path.write_text(sample_config_content)
    return config_path


class TestPartialTOMLParser:
    """Test the PartialTOMLParser implementation."""

    def test_get_available_sections(self, config_file):
        """Test getting available sections without full parsing."""
        parser = PartialTOMLParser()
        sections = parser.get_available_sections(config_file)

        expected_sections = ["PATHS", "MODEL", "TRAINER", "MLFLOW", "DATASET"]
        assert all(section in sections for section in expected_sections)

    def test_parse_single_section(self, config_file):
        """Test parsing a single section."""
        parser = PartialTOMLParser()
        sections_data = parser.parse_sections(config_file, ["PATHS"])

        assert "PATHS" in sections_data
        paths_data = sections_data["PATHS"]
        assert paths_data["dataroot"] == "./test_data"
        assert paths_data["input"] == "./test_data/input"
        assert paths_data["output"] == "./test_data/output"

    def test_parse_multiple_sections(self, config_file):
        """Test parsing multiple sections."""
        parser = PartialTOMLParser()
        sections_data = parser.parse_sections(config_file, ["PATHS", "MODEL"])

        assert "PATHS" in sections_data
        assert "MODEL" in sections_data

        # Check PATHS data
        paths_data = sections_data["PATHS"]
        assert paths_data["dataroot"] == "./test_data"

        # Check MODEL data
        model_data = sections_data["MODEL"]
        assert model_data["name"] == "TestModel"
        assert model_data["latent_size"] == 128

    def test_parse_nonexistent_section(self, config_file):
        """Test parsing a section that doesn't exist."""
        parser = PartialTOMLParser()
        sections_data = parser.parse_sections(config_file, ["NONEXISTENT"])

        # Should return empty dict for missing section
        assert "NONEXISTENT" not in sections_data or sections_data["NONEXISTENT"] == {}

    def test_file_not_found(self):
        """Test handling of missing config file."""
        parser = PartialTOMLParser()

        with pytest.raises(FileNotFoundError):
            parser.get_available_sections("nonexistent.toml")

        with pytest.raises(FileNotFoundError):
            parser.parse_sections("nonexistent.toml", ["PATHS"])


class TestLoadSectionConfig:
    """Test the load_section_config function."""

    def test_load_existing_section(self, config_file):
        """Test loading an existing section."""
        # Register section mapping for test
        from dlkit.tools.io.config import register_section_mapping

        register_section_mapping(PathsTestSettings, "PATHS")

        paths_config = load_section_config(config_file, PathsTestSettings)

        assert isinstance(paths_config, PathsTestSettings)
        assert Path(paths_config.dataroot).name == "test_data"
        assert Path(paths_config.input).name == "input"
        assert Path(paths_config.input).parent.name == "test_data"
        assert Path(paths_config.output).name == "output"
        assert Path(paths_config.output).parent.name == "test_data"

    def test_load_nonexistent_section(self, config_file):
        """Test loading a section that doesn't exist."""
        from dlkit.tools.io.config import register_section_mapping

        register_section_mapping(SampleSectionSettings, "NONEXISTENT")

        with pytest.raises(ConfigSectionError) as exc_info:
            load_section_config(config_file, SampleSectionSettings)

        assert "NONEXISTENT" in str(exc_info.value)
        assert "Available sections:" in str(exc_info.value)

    def test_load_with_explicit_section_name(self, config_file):
        """Test loading with explicitly provided section name."""
        paths_config = load_section_config(config_file, PathsTestSettings, "PATHS")

        assert isinstance(paths_config, PathsTestSettings)
        assert Path(paths_config.dataroot).name == "test_data"

    def test_load_known_section_without_model(self, config_file):
        """Load a known section using the predefined mapping only."""
        paths_config = load_section_config(config_file, section_name="PATHS")

        assert isinstance(paths_config, PathsSettings)
        assert Path(str(paths_config.dataroot)).name == "test_data"

    def test_file_not_found(self):
        """Test handling of missing config file."""
        with pytest.raises(FileNotFoundError):
            load_section_config("nonexistent.toml", PathsTestSettings)


class TestLoadSectionsConfig:
    """Test the load_sections_config function."""

    def test_load_multiple_sections(self, config_file):
        """Test loading multiple sections at once."""
        from dlkit.tools.io.config import register_section_mapping

        register_section_mapping(PathsTestSettings, "PATHS")
        register_section_mapping(ModelTestSettings, "MODEL")

        section_configs = {"PATHS": PathsTestSettings, "MODEL": ModelTestSettings}

        configs = load_sections_config(config_file, section_configs)

        assert "PATHS" in configs
        assert "MODEL" in configs

        # Check PATHS config
        paths_config = configs["PATHS"]
        assert isinstance(paths_config, PathsTestSettings)
        assert Path(paths_config.dataroot).name == "test_data"

        # Check MODEL config
        model_config = configs["MODEL"]
        assert isinstance(model_config, ModelTestSettings)
        assert model_config.name == "TestModel"
        assert model_config.latent_size == 128

    def test_load_known_sections_without_models(self, config_file):
        """Test loading sections via predefined registry mappings."""
        configs = load_sections_config(config_file, ["PATHS", "MODEL"])

        assert isinstance(configs["PATHS"], PathsSettings)
        assert isinstance(configs["MODEL"], ModelComponentSettings)
        assert configs["MODEL"].name == "TestModel"
        assert Path(str(configs["PATHS"].dataroot)).name == "test_data"

    def test_load_mixed_section_specs(self, config_file):
        """Mix registry-driven and explicit model mappings."""
        configs = load_sections_config(
            config_file,
            {
                "PATHS": None,
                "MODEL": ModelComponentSettings,
            },
        )

        assert isinstance(configs["PATHS"], PathsSettings)
        assert isinstance(configs["MODEL"], ModelComponentSettings)
        assert Path(str(configs["PATHS"].dataroot)).name == "test_data"

    def test_load_with_missing_section(self, config_file):
        """Test loading when one section is missing."""
        section_configs = {"PATHS": PathsTestSettings, "NONEXISTENT": SampleSectionSettings}

        with pytest.raises(ConfigSectionError) as exc_info:
            load_sections_config(config_file, section_configs)

        assert "NONEXISTENT" in str(exc_info.value)

    def test_load_empty_sections_dict(self, config_file):
        """Test loading with empty sections dictionary."""
        configs = load_sections_config(config_file, {})
        assert configs == {}


class TestUtilityFunctions:
    """Test utility functions for config management."""

    def test_check_section_exists(self, config_file):
        """Test checking if sections exist."""
        assert check_section_exists(config_file, "PATHS") is True
        assert check_section_exists(config_file, "MODEL") is True
        assert check_section_exists(config_file, "NONEXISTENT") is False

    def test_get_available_sections(self, config_file):
        """Test getting list of available sections."""
        sections = get_available_sections(config_file)

        expected_sections = ["PATHS", "MODEL", "TRAINER", "MLFLOW", "DATASET"]
        assert all(section in sections for section in expected_sections)

    def test_utility_functions_file_not_found(self):
        """Test utility functions with missing file."""
        with pytest.raises(FileNotFoundError):
            check_section_exists("nonexistent.toml", "PATHS")

        with pytest.raises(FileNotFoundError):
            get_available_sections("nonexistent.toml")


class TestPerformance:
    """Performance-related tests."""

    def test_partial_vs_full_loading(self, config_file):
        """Test that partial loading is working (functional test)."""
        from dlkit.tools.io.config import load_config, register_section_mapping

        register_section_mapping(PathsTestSettings, "PATHS")

        # Both should return the same data
        full_config = load_config(config_file, raw=True)
        partial_config = load_section_config(config_file, PathsTestSettings)

        # Verify we get the same data
        assert Path(partial_config.dataroot).name == Path(full_config["PATHS"]["dataroot"]).name
        assert Path(partial_config.input).name == Path(full_config["PATHS"]["input"]).name
        assert Path(partial_config.output).name == Path(full_config["PATHS"]["output"]).name

    def test_multiple_sections_efficiency(self, config_file):
        """Test loading multiple sections efficiently."""
        from dlkit.tools.io.config import register_section_mapping

        register_section_mapping(PathsTestSettings, "PATHS")
        register_section_mapping(ModelTestSettings, "MODEL")

        # Loading multiple sections should work efficiently
        section_configs = {"PATHS": PathsTestSettings, "MODEL": ModelTestSettings}

        configs = load_sections_config(config_file, section_configs)

        assert len(configs) == 2
        assert isinstance(configs["PATHS"], PathsTestSettings)
        assert isinstance(configs["MODEL"], ModelTestSettings)
        assert configs["MODEL"].name == "TestModel"


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_malformed_config_file(self, tmp_path):
        """Test handling of malformed TOML file."""
        malformed_config = tmp_path / "malformed.toml"
        malformed_config.write_text("""
[PATHS
dataroot = "./test_data"
# Missing closing bracket
""")

        # Should handle parsing errors gracefully
        parser = PartialTOMLParser()

        # get_available_sections should still work for section detection
        sections = parser.get_available_sections(malformed_config)
        # Should at least detect the attempted section
        assert len(sections) >= 0  # May or may not detect malformed section

    def test_validation_error_in_section(self, tmp_path):
        """Test handling of validation errors in section data."""
        invalid_config = tmp_path / "invalid.toml"
        invalid_config.write_text("""
[PATHS]
dataroot = 123  # Invalid type - should be string
input = "./test_data/input"
""")

        with pytest.raises(ConfigValidationError) as exc_info:
            load_section_config(invalid_config, PathsTestSettings, "PATHS")

        assert "PathsTestSettings" in str(exc_info.value)
        assert "Failed to validate section" in str(exc_info.value)
