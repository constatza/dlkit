"""Tests for config parsing protocols and implementations."""

import pytest
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field

from dlkit.tools.io.protocols import (
    ConfigParser,
    SectionExtractor,
    ConfigValidator,
    PartialConfigReader,
)
from dlkit.tools.io.parsers import (
    PartialTOMLParser,
    DynafconfConfigParser,
    StandardSectionExtractor,
)


class SimpleTestSettings(BaseModel):
    """Simple test settings for protocol testing."""

    name: str = Field(default="test")
    value: int = Field(default=1)


class ConfigValidatorImpl:
    """Implementation of ConfigValidator protocol for testing."""

    def validate_section[T: BaseModel](
        self, section_data: dict[str, Any], model_class: type[T]
    ) -> T:
        """Validate section data with Pydantic model."""
        return model_class.model_validate(section_data)

    def validate_sections(
        self, sections_data: dict[str, dict[str, Any]], model_classes: dict[str, type[BaseModel]]
    ) -> dict[str, BaseModel]:
        """Validate multiple sections."""
        validated = {}
        for section_name, section_data in sections_data.items():
            if section_name in model_classes:
                model_class = model_classes[section_name]
                validated[section_name] = self.validate_section(section_data, model_class)
        return validated


@pytest.fixture
def sample_config_file(tmp_path):
    """Create a sample config file for testing."""
    config_content = """
[SECTION1]
name = "test1"
value = 42

[SECTION2]
name = "test2"
value = 100

[NESTED.SECTION]
name = "nested"
value = 200
"""
    config_path = tmp_path / "test_config.toml"
    config_path.write_text(config_content)
    return config_path


class TestConfigParserProtocol:
    """Test ConfigParser protocol implementations."""

    def test_partial_toml_parser_implements_protocol(self):
        """Test that PartialTOMLParser implements ConfigParser protocol."""
        parser = PartialTOMLParser()
        assert isinstance(parser, ConfigParser)

    def test_dynaconf_parser_implements_protocol(self):
        """Test that DynafconfConfigParser implements ConfigParser protocol."""
        parser = DynafconfConfigParser()
        assert isinstance(parser, ConfigParser)

    def test_parse_full_method(self, sample_config_file):
        """Test parse_full method."""
        parser = PartialTOMLParser()
        config_data = parser.parse_full(sample_config_file)

        assert isinstance(config_data, dict)
        assert "SECTION1" in config_data
        assert "SECTION2" in config_data

    def test_parse_sections_method(self, sample_config_file):
        """Test parse_sections method."""
        parser = PartialTOMLParser()
        sections_data = parser.parse_sections(sample_config_file, ["SECTION1"])

        assert isinstance(sections_data, dict)
        assert "SECTION1" in sections_data
        assert sections_data["SECTION1"]["name"] == "test1"

    def test_get_available_sections_method(self, sample_config_file):
        """Test get_available_sections method."""
        parser = PartialTOMLParser()
        sections = parser.get_available_sections(sample_config_file)

        assert isinstance(sections, list)
        assert "SECTION1" in sections
        assert "SECTION2" in sections

    def test_file_not_found_handling(self):
        """Test that implementations handle missing files correctly."""
        parser = PartialTOMLParser()

        with pytest.raises(FileNotFoundError):
            parser.parse_full("nonexistent.toml")

        with pytest.raises(FileNotFoundError):
            parser.parse_sections("nonexistent.toml", ["SECTION1"])

        with pytest.raises(FileNotFoundError):
            parser.get_available_sections("nonexistent.toml")


class TestSectionExtractorProtocol:
    """Test SectionExtractor protocol implementation."""

    def test_standard_extractor_implements_protocol(self):
        """Test that StandardSectionExtractor implements SectionExtractor protocol."""
        extractor = StandardSectionExtractor()
        assert isinstance(extractor, SectionExtractor)

    def test_extract_section_method(self):
        """Test extract_section method."""
        extractor = StandardSectionExtractor()
        config_data = {
            "SECTION1": {"name": "test1", "value": 42},
            "SECTION2": {"name": "test2", "value": 100},
        }

        section_data = extractor.extract_section(config_data, "SECTION1")
        assert section_data == {"name": "test1", "value": 42}

        # Test missing section
        missing_data = extractor.extract_section(config_data, "MISSING")
        assert missing_data is None

    def test_extract_sections_method(self):
        """Test extract_sections method."""
        extractor = StandardSectionExtractor()
        config_data = {
            "SECTION1": {"name": "test1", "value": 42},
            "SECTION2": {"name": "test2", "value": 100},
            "SECTION3": {"name": "test3", "value": 300},
        }

        sections_data = extractor.extract_sections(config_data, ["SECTION1", "SECTION3", "MISSING"])

        assert "SECTION1" in sections_data
        assert "SECTION3" in sections_data
        assert "MISSING" not in sections_data  # Missing sections not included
        assert sections_data["SECTION1"] == {"name": "test1", "value": 42}
        assert sections_data["SECTION3"] == {"name": "test3", "value": 300}


class TestConfigValidatorProtocol:
    """Test ConfigValidator protocol implementation."""

    def test_validator_implements_protocol(self):
        """Test that our validator implements ConfigValidator protocol."""
        validator = ConfigValidatorImpl()
        assert isinstance(validator, ConfigValidator)

    def test_validate_section_method(self):
        """Test validate_section method."""
        validator = ConfigValidatorImpl()
        section_data = {"name": "test", "value": 42}

        validated = validator.validate_section(section_data, SimpleTestSettings)
        assert isinstance(validated, SimpleTestSettings)
        assert validated.name == "test"
        assert validated.value == 42

    def test_validate_sections_method(self):
        """Test validate_sections method."""
        validator = ConfigValidatorImpl()
        sections_data = {
            "SECTION1": {"name": "test1", "value": 42},
            "SECTION2": {"name": "test2", "value": 100},
        }
        model_classes = {"SECTION1": SimpleTestSettings, "SECTION2": SimpleTestSettings}

        validated = validator.validate_sections(sections_data, model_classes)

        assert "SECTION1" in validated
        assert "SECTION2" in validated
        assert isinstance(validated["SECTION1"], SimpleTestSettings)
        assert isinstance(validated["SECTION2"], SimpleTestSettings)
        assert validated["SECTION1"].name == "test1"
        assert validated["SECTION2"].name == "test2"

    def test_validation_error_handling(self):
        """Test that validation errors are properly raised."""
        validator = ConfigValidatorImpl()
        invalid_data = {"name": 123, "value": "not_an_int"}  # Wrong types

        with pytest.raises(Exception):  # Pydantic ValidationError
            validator.validate_section(invalid_data, SimpleTestSettings)


class TestProtocolIntegration:
    """Test how protocols work together."""

    def test_parser_extractor_validator_integration(self, sample_config_file):
        """Test integration between parser, extractor, and validator."""
        # Parse config
        parser = PartialTOMLParser()
        config_data = parser.parse_full(sample_config_file)

        # Extract sections
        extractor = StandardSectionExtractor()
        section_data = extractor.extract_section(config_data, "SECTION1")

        # Validate section
        validator = ConfigValidatorImpl()
        validated = validator.validate_section(section_data, SimpleTestSettings)

        assert isinstance(validated, SimpleTestSettings)
        assert validated.name == "test1"
        assert validated.value == 42

    def test_end_to_end_multi_section_workflow(self, sample_config_file):
        """Test end-to-end workflow with multiple sections."""
        # Parse only specific sections
        parser = PartialTOMLParser()
        sections_data = parser.parse_sections(sample_config_file, ["SECTION1", "SECTION2"])

        # Validate all sections
        validator = ConfigValidatorImpl()
        model_classes = {"SECTION1": SimpleTestSettings, "SECTION2": SimpleTestSettings}
        validated = validator.validate_sections(sections_data, model_classes)

        assert len(validated) == 2
        assert validated["SECTION1"].name == "test1"
        assert validated["SECTION1"].value == 42
        assert validated["SECTION2"].name == "test2"
        assert validated["SECTION2"].value == 100


class TestPartialConfigReaderProtocol:
    """Test the high-level PartialConfigReader protocol."""

    class PartialConfigReaderImpl:
        """Implementation of PartialConfigReader for testing."""

        def __init__(self):
            self.parser = PartialTOMLParser()
            self.validator = ConfigValidatorImpl()

        def read_section[U: BaseModel](self, config_path: Path | str, model_class: type[U]) -> U:
            """Read and validate a single section."""
            # Get section name (simplified)
            section_name = model_class.__name__.replace("Settings", "").upper()

            config_data = self.parser.parse_full(config_path)
            if section_name not in config_data:
                raise ValueError(f"Section {section_name} not found")

            section_data = config_data[section_name]
            return self.validator.validate_section(section_data, model_class)

        def read_sections(
            self, config_path: Path | str, section_configs: dict[str, type[BaseModel]]
        ) -> dict[str, BaseModel]:
            """Read and validate multiple sections."""
            sections_data = self.parser.parse_sections(config_path, list(section_configs.keys()))
            return self.validator.validate_sections(sections_data, section_configs)

        def section_exists(self, config_path: Path | str, section_name: str) -> bool:
            """Check if section exists."""
            available = self.parser.get_available_sections(config_path)
            return section_name in available

    def test_reader_implements_protocol(self):
        """Test that our reader implements PartialConfigReader protocol."""
        reader = self.PartialConfigReaderImpl()
        assert isinstance(reader, PartialConfigReader)

    def test_read_section_method(self, sample_config_file):
        """Test read_section method."""
        reader = self.PartialConfigReaderImpl()

        # Create a test config with appropriate section name
        test_config = sample_config_file.parent / "simple_test.toml"
        test_config.write_text("""
[SIMPLETEST]
name = "test"
value = 42
""")

        validated = reader.read_section(test_config, SimpleTestSettings)
        assert isinstance(validated, SimpleTestSettings)
        assert validated.name == "test"
        assert validated.value == 42

    def test_read_sections_method(self, sample_config_file):
        """Test read_sections method."""
        reader = self.PartialConfigReaderImpl()

        section_configs = {"SECTION1": SimpleTestSettings, "SECTION2": SimpleTestSettings}

        validated = reader.read_sections(sample_config_file, section_configs)

        assert len(validated) == 2
        assert isinstance(validated["SECTION1"], SimpleTestSettings)
        assert isinstance(validated["SECTION2"], SimpleTestSettings)

    def test_section_exists_method(self, sample_config_file):
        """Test section_exists method."""
        reader = self.PartialConfigReaderImpl()

        assert reader.section_exists(sample_config_file, "SECTION1") is True
        assert reader.section_exists(sample_config_file, "SECTION2") is True
        assert reader.section_exists(sample_config_file, "MISSING") is False


class TestProtocolCompliance:
    """Test that implementations properly follow protocol contracts."""

    def test_parser_protocol_methods_exist(self):
        """Test that parser implementations have all required methods."""
        parsers = [PartialTOMLParser(), DynafconfConfigParser()]

        for parser in parsers:
            assert hasattr(parser, "parse_full")
            assert hasattr(parser, "parse_sections")
            assert hasattr(parser, "get_available_sections")
            assert callable(getattr(parser, "parse_full"))
            assert callable(getattr(parser, "parse_sections"))
            assert callable(getattr(parser, "get_available_sections"))

    def test_extractor_protocol_methods_exist(self):
        """Test that extractor implementation has all required methods."""
        extractor = StandardSectionExtractor()

        assert hasattr(extractor, "extract_section")
        assert hasattr(extractor, "extract_sections")
        assert callable(getattr(extractor, "extract_section"))
        assert callable(getattr(extractor, "extract_sections"))

    def test_validator_protocol_methods_exist(self):
        """Test that validator implementation has all required methods."""
        validator = ConfigValidatorImpl()

        assert hasattr(validator, "validate_section")
        assert hasattr(validator, "validate_sections")
        assert callable(getattr(validator, "validate_section"))
        assert callable(getattr(validator, "validate_sections"))
