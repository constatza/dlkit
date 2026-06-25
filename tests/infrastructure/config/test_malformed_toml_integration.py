"""Integration tests for malformed TOML configurations.

Verifies that malformed configs are caught early (fail-fast) rather than
failing late during dataset instantiation or training.
"""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from dlkit.infrastructure.config.data_settings import DataSettings
from dlkit.infrastructure.config.factories import load_job

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_data_file(tmp_path: Path) -> Path:
    data_file = tmp_path / "features.npy"
    np.save(data_file, np.ones((10, 5), dtype=np.float32))
    return data_file


@pytest.fixture
def sample_target_file(tmp_path: Path) -> Path:
    target_file = tmp_path / "targets.npy"
    np.save(target_file, np.ones((10, 1), dtype=np.float32))
    return target_file


# ============================================================================
# Missing Name Field Tests (Production Bug)
# ============================================================================


class TestMissingNameInTOML:
    """Missing 'name' in [[data.features]] or [[data.targets]] must fail at validation."""

    def test_missing_name_in_features_fails_at_config_load(self, tmp_path: Path) -> None:
        data_path = tmp_path / "features.npy"
        with pytest.raises((ValidationError, Exception)):
            DataSettings.model_validate(
                {
                    "class": "FlexibleDataset",
                    "features": [{"format": "npy", "path": str(data_path)}],
                }
            )

    def test_missing_name_in_targets_fails_at_config_load(self, tmp_path: Path) -> None:
        data_path = tmp_path / "targets.npy"
        with pytest.raises((ValidationError, Exception)):
            DataSettings.model_validate(
                {
                    "class": "FlexibleDataset",
                    "targets": [{"format": "npy", "path": str(data_path)}],
                }
            )

    def test_multiple_entries_one_missing_name_fails(self, tmp_path: Path) -> None:
        x_path = tmp_path / "x.npy"
        z_path = tmp_path / "z.npy"
        with pytest.raises((ValidationError, Exception)):
            DataSettings.model_validate(
                {
                    "class": "FlexibleDataset",
                    "features": [
                        {"name": "x", "format": "npy", "path": str(x_path)},
                        {"format": "npy", "path": str(tmp_path / "y.npy")},
                    ],
                    "targets": [{"name": "z", "format": "npy", "path": str(z_path)}],
                }
            )


# ============================================================================
# Valid Configuration Tests
# ============================================================================


class TestValidConfigurations:
    """Valid configurations load correctly."""

    def test_valid_features_and_targets_succeed(
        self, sample_data_file: Path, sample_target_file: Path
    ) -> None:
        ds = DataSettings.model_validate(
            {
                "class": "FlexibleDataset",
                "features": [{"name": "x", "format": "npy", "path": str(sample_data_file)}],
                "targets": [{"name": "y", "format": "npy", "path": str(sample_target_file)}],
            }
        )
        assert len(ds.features) == 1
        assert len(ds.targets) == 1
        assert ds.features[0].name == "x"
        assert ds.targets[0].name == "y"

    def test_multiple_features_all_with_names_succeed(self, tmp_path: Path) -> None:
        for name in ("x1.npy", "x2.npy", "x3.npy", "y.npy"):
            np.save(tmp_path / name, np.random.rand(10, 5))

        ds = DataSettings.model_validate(
            {
                "class": "FlexibleDataset",
                "features": [
                    {"name": "x1", "format": "npy", "path": str(tmp_path / "x1.npy")},
                    {"name": "x2", "format": "npy", "path": str(tmp_path / "x2.npy")},
                    {"name": "x3", "format": "npy", "path": str(tmp_path / "x3.npy")},
                ],
                "targets": [{"name": "y", "format": "npy", "path": str(tmp_path / "y.npy")}],
            }
        )
        assert len(ds.features) == 3
        assert all(f.name for f in ds.features)


class TestSectionLevelOptionality:
    """Partial configs are supported — only required sections must be present."""

    def test_training_config_loads_without_optional_sections(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text("""
[run]
type = "train"

[experiment]
name = "section_only"

[model]
class = "ConstantWidthFFNN"
module_path = "dlkit.domain.nn"

[data]
class = "FlexibleDataset"

[training.trainer]
max_epochs = 1

[training.optimizer.default_optimizer]
name = "Adam"
lr = 0.001

[training.loss]
name = "MSELoss"
module_path = "torch.nn"
""")
        job = load_job(config)
        assert job.experiment is not None
        assert job.experiment.name == "section_only"
        assert job.training is not None
        assert job.search is None
        assert job.tracking.backend == "none"


class TestWorkflowLoaderParseErrors:
    """Workflow loading surfaces parse failures."""

    def test_load_job_raises_for_malformed_toml(self, tmp_path: Path) -> None:
        config = tmp_path / "malformed.toml"
        config.write_text("""
[run
type = "train"
""")
        with pytest.raises(Exception):
            load_job(config)
