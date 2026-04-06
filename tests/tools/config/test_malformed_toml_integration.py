"""Integration tests for malformed TOML configurations.

This module tests the end-to-end flow: TOML → Settings → Dataset
to verify that malformed configs are caught early (fail-fast principle)
rather than failing late during dataset instantiation or training.

These tests verify the fix for the production bug where configs with
missing 'name' fields in [[DATASET.features]] or [[DATASET.targets]]
passed validation but failed at runtime.
"""

from pathlib import Path
from typing import cast

import numpy as np
import pytest
from pydantic import BaseModel

from dlkit.infrastructure.config.dataset_settings import DatasetSettings
from dlkit.infrastructure.io.config import (
    ConfigValidationError,
    load_sections_config,
    load_training_config_eager,
)


def _dataset_section(settings: dict[str, BaseModel]) -> DatasetSettings:
    """Narrow the DATASET section from generic section-loading output."""
    return cast("DatasetSettings", settings["DATASET"])


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_data_file(tmp_path: Path) -> Path:
    """Create a sample .npy data file for testing."""
    data_file = tmp_path / "features.npy"
    np.save(data_file, np.ones((10, 5), dtype=np.float32))
    return data_file


@pytest.fixture
def sample_target_file(tmp_path: Path) -> Path:
    """Create a sample .npy target file for testing."""
    target_file = tmp_path / "targets.npy"
    np.save(target_file, np.ones((10, 1), dtype=np.float32))
    return target_file


# ============================================================================
# Missing Name Field Tests (Production Bug)
# ============================================================================


class TestMissingNameInTOML:
    """Test that missing 'name' field in TOML features/targets fails early.

    These tests verify the fix for the production bug where malformed
    configs passed load_sections_config() but failed later.
    """

    def test_missing_name_in_features_fails_at_config_load(self, tmp_path: Path):
        """TOML with missing 'name' in [[DATASET.features]] should fail early."""
        config = tmp_path / "config.toml"
        data_path = tmp_path / "features.npy"
        config.write_text(f"""
[DATASET]
name = "FlexibleDataset"

[[DATASET.features]]
path = "{data_path.as_posix()}"
""")

        with pytest.raises(ConfigValidationError, match="requires 'name'"):
            load_sections_config(config, ["DATASET"])

    def test_missing_name_in_targets_fails_at_config_load(self, tmp_path: Path):
        """TOML with missing 'name' in [[DATASET.targets]] should fail early."""
        config = tmp_path / "config.toml"
        data_path = tmp_path / "targets.npy"
        config.write_text(f"""
[DATASET]
name = "FlexibleDataset"

[[DATASET.targets]]
path = "{data_path.as_posix()}"
""")

        with pytest.raises(ConfigValidationError, match="requires 'name'"):
            load_sections_config(config, ["DATASET"])

    def test_missing_name_error_message_includes_toml_example(self, tmp_path: Path):
        """Error message should include helpful TOML fix example."""
        config = tmp_path / "config.toml"
        data_path = tmp_path / "test.npy"
        config.write_text(f"""
[[DATASET.features]]
path = "{data_path.as_posix()}"
""")

        with pytest.raises(ConfigValidationError) as exc_info:
            load_sections_config(config, ["DATASET"])

        error_msg = str(exc_info.value)
        assert "DATASET.features" in error_msg or "requires 'name'" in error_msg

    def test_multiple_entries_one_missing_name_fails(self, tmp_path: Path):
        """If one entry missing name, config should fail even if others valid."""
        config = tmp_path / "config.toml"
        x_path = tmp_path / "x.npy"
        y_path = tmp_path / "y.npy"
        z_path = tmp_path / "z.npy"
        config.write_text(f"""
[[DATASET.features]]
name = "x"
path = "{x_path.as_posix()}"

[[DATASET.features]]
# Missing name here!
path = "{y_path.as_posix()}"

[[DATASET.targets]]
name = "z"
path = "{z_path.as_posix()}"
""")

        with pytest.raises(ConfigValidationError, match="requires 'name'"):
            load_sections_config(config, ["DATASET"])


# ============================================================================
# Valid Configuration Tests
# ============================================================================


class TestValidConfigurations:
    """Test that valid configurations continue to work correctly."""

    def test_valid_features_and_targets_succeed(
        self, tmp_path: Path, sample_data_file: Path, sample_target_file: Path
    ):
        """Valid TOML with name + path should succeed."""
        config = tmp_path / "config.toml"
        config.write_text(f"""
[DATASET]
name = "FlexibleDataset"

[[DATASET.features]]
name = "x"
path = "{sample_data_file.as_posix()}"

[[DATASET.targets]]
name = "y"
path = "{sample_target_file.as_posix()}"
""")

        settings = _dataset_section(load_sections_config(config, ["DATASET"]))
        assert len(settings.features) == 1
        assert len(settings.targets) == 1
        assert settings.features[0].name == "x"
        assert settings.targets[0].name == "y"

    def test_multiple_features_all_with_names_succeed(self, tmp_path: Path):
        """Multiple features all with names should succeed."""
        # Create actual data files
        x1_path = tmp_path / "x1.npy"
        x2_path = tmp_path / "x2.npy"
        x3_path = tmp_path / "x3.npy"
        y_path = tmp_path / "y.npy"
        np.save(x1_path, np.random.rand(10, 5))
        np.save(x2_path, np.random.rand(10, 5))
        np.save(x3_path, np.random.rand(10, 5))
        np.save(y_path, np.random.rand(10, 1))

        config = tmp_path / "config.toml"
        config.write_text(f"""
[DATASET]
name = "FlexibleDataset"

[[DATASET.features]]
name = "x1"
path = "{x1_path.as_posix()}"

[[DATASET.features]]
name = "x2"
path = "{x2_path.as_posix()}"

[[DATASET.features]]
name = "x3"
path = "{x3_path.as_posix()}"

[[DATASET.targets]]
name = "y"
path = "{y_path.as_posix()}"
""")

        settings = _dataset_section(load_sections_config(config, ["DATASET"]))
        assert len(settings.features) == 3
        assert all(f.name for f in settings.features)


class TestSectionLevelOptionality:
    """Ensure partial configs are only supported at whole-section granularity."""

    def test_training_config_loads_without_optional_sections(self, tmp_path: Path):
        """SESSION and TRAINING alone should load; other sections remain None."""
        config = tmp_path / "config.toml"
        config.write_text("""
[SESSION]
name = "section_only"

[TRAINING]
epochs = 1

[TRAINING.optimizer]
name = "Adam"
lr = 0.001

[TRAINING.loss_function]
name = "MSELoss"
module_path = "torch.nn"
""")

        config_obj = load_training_config_eager(config)
        assert config_obj.SESSION.name == "section_only"
        assert config_obj.TRAINING is not None
        assert config_obj.DATAMODULE is None
        assert config_obj.DATASET is None
        assert config_obj.MODEL is None
