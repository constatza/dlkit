"""Tests for eager validation with the new JobConfig hierarchy.

This module tests:
1. JobConfig validation fails fast on bad values
2. DataSettings validates paths eagerly
3. Session precision aliases work via RunSettings
4. Completeness validators ensure configs are build-ready
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from dlkit.infrastructure.config.data_settings import DataSettings
from dlkit.infrastructure.config.model_components import (
    LossComponentSettings,
    MetricComponentSettings,
    ModelComponentSettings,
)
from dlkit.infrastructure.config.optimizer_settings import OptimizerSettings
from dlkit.infrastructure.config.run_settings import RunSettings
from dlkit.infrastructure.config.validators import (
    ConfigValidationError,
    validate_inference_config_complete,
    validate_training_config_complete,
)
from dlkit.infrastructure.precision import PrecisionStrategy

# ============================================================================
# Failure Cases: Invalid Configs Fail at Load Time
# ============================================================================


class TestEagerValidationFailureCases:
    """Test that invalid configs fail immediately at load time."""

    def test_invalid_path_fails_immediately_at_load_time(self):
        """Test that configs with invalid paths fail at load (not at build time)."""
        from dlkit.infrastructure.config.job_config import TrainingJobConfig

        config_dict = {
            "run": {"type": "train", "seed": 42},
            "model": {"class": "LinearNetwork", "module_path": "dlkit.domain.nn.ffnn"},
            "data": {
                "class": "FlexibleDataset",
                "features": [
                    {"name": "x", "format": "npy", "path": "/this/path/does/not/exist.npy"},
                ],
            },
            "training": {
                "trainer": {"max_epochs": 10},
                "optimizer": {"default_optimizer": {"name": "Adam", "lr": 0.001}},
                "loss": {"name": "MSELoss", "module_path": "torch.nn"},
            },
        }

        with pytest.raises(ValidationError) as exc_info:
            TrainingJobConfig.model_validate(config_dict)

        error_msg = str(exc_info.value)
        assert "path" in error_msg.lower() or "exist" in error_msg.lower()

    def test_type_error_fails_immediately(self):
        """Test that type errors fail immediately at load time."""
        from dlkit.infrastructure.config.job_config import TrainingJobConfig

        config_dict = {
            "run": {"type": "train", "seed": 42},
            "model": {"class": "LinearNetwork", "module_path": "dlkit.domain.nn.ffnn"},
            "data": {"class": "FlexibleDataset"},
            "training": {
                "trainer": {"max_epochs": "not_an_integer"},  # Wrong type!
                "optimizer": {"default_optimizer": {"name": "Adam", "lr": 0.001}},
                "loss": {"name": "MSELoss", "module_path": "torch.nn"},
            },
        }

        with pytest.raises(ValidationError):
            TrainingJobConfig.model_validate(config_dict)

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: ModelComponentSettings(
                name="BrokenModel",
                module_path="dlkit.not_a_real_module",
            ),
            lambda: OptimizerSettings(module_path="dlkit.not_a_real_module"),
            lambda: LossComponentSettings(module_path="dlkit.not_a_real_module"),
            lambda: MetricComponentSettings(module_path="dlkit.not_a_real_module"),
        ],
    )
    def test_invalid_module_paths_fail_at_load_time(self, factory) -> None:
        with pytest.raises(ValidationError, match="module_path"):
            factory()


class TestImportIsolation:
    """Ensure non-graph workflows do not import graph backends eagerly."""

    @staticmethod
    def _run_warning_error_import(code: str) -> subprocess.CompletedProcess[str]:
        repo_root = Path(__file__).resolve().parents[3]
        env = os.environ.copy()
        pythonpath = str(repo_root / "src")
        env["PYTHONPATH"] = (
            pythonpath
            if not env.get("PYTHONPATH")
            else f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
        )
        env["MPLCONFIGDIR"] = "/tmp/matplotlib"
        return subprocess.run(
            [sys.executable, "-W", "error", "-c", code],
            cwd=repo_root,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_importing_transforms_base_is_graph_free_under_warning_error(self) -> None:
        result = self._run_warning_error_import("import dlkit.domain.transforms.base")
        assert result.returncode == 0, result.stderr

    def test_importing_broad_dataset_namespace_is_graph_free_under_warning_error(self) -> None:
        result = self._run_warning_error_import(
            "from dlkit.engine.data.datasets import FlexibleDataset"
        )
        assert result.returncode == 0, result.stderr

    def test_importing_broad_datamodule_namespace_is_graph_free_under_warning_error(self) -> None:
        result = self._run_warning_error_import(
            "from dlkit.engine.adapters.lightning.datamodules import ArrayDataModule"
        )
        assert result.returncode == 0, result.stderr

    def test_validating_non_graph_training_config_is_warning_clean(self) -> None:
        code = """
from pathlib import Path
import tempfile
import numpy as np
from dlkit.infrastructure.config.job_config import TrainingJobConfig

with tempfile.TemporaryDirectory() as tmp_dir:
    tmp = Path(tmp_dir)
    features = tmp / "features.npy"
    targets = tmp / "targets.npy"
    np.save(features, np.random.rand(8, 4))
    np.save(targets, np.random.rand(8, 1))
    TrainingJobConfig.model_validate(
        {
            "run": {"type": "train", "seed": 1},
            "model": {"class": "LinearNetwork", "module_path": "dlkit.domain.nn.ffnn"},
            "data": {
                "class": "FlexibleDataset",
                "batch_size": 2,
                "features": [{"name": "x", "format": "npy", "path": str(features)}],
                "targets": [{"name": "y", "format": "npy", "path": str(targets)}],
            },
            "training": {
                "trainer": {"max_epochs": 1, "accelerator": "cpu"},
                "optimizer": {"default_optimizer": {"name": "Adam", "lr": 0.001}},
                "loss": {"name": "MSELoss", "module_path": "torch.nn"},
            },
        }
    )
"""
        result = self._run_warning_error_import(code)
        assert result.returncode == 0, result.stderr

    def test_validating_non_graph_inference_config_is_warning_clean(self) -> None:
        code = """
from pathlib import Path
import tempfile
from dlkit.infrastructure.config.job_config import InferenceJobConfig

with tempfile.TemporaryDirectory() as tmp_dir:
    checkpoint = Path(tmp_dir) / "model.ckpt"
    checkpoint.write_text("fake checkpoint")
    InferenceJobConfig.model_validate(
        {
            "run": {"type": "predict", "seed": 1},
            "model": {
                "class": "LinearNetwork",
                "module_path": "dlkit.domain.nn.ffnn",
                "checkpoint": str(checkpoint),
            },
        }
    )
"""
        result = self._run_warning_error_import(code)
        assert result.returncode == 0, result.stderr


class TestRunSettingsPrecisionAliases:
    """Ensure run.precision accepts Lightning-style aliases."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (32, "32"),
            ("32", "32"),
            (16, "16"),
            ("16", "16"),
            ("16-mixed", "16-mixed"),
            (64, "64"),
            ("64", "64"),
            ("bf16", "bf16"),
            ("bf16-mixed", "bf16-mixed"),
        ],
    )
    def test_precision_aliases_are_normalized(self, value: object, expected: str) -> None:
        settings = RunSettings.model_validate({"precision": value})
        assert isinstance(settings.precision, PrecisionStrategy)
        assert str(settings.precision) == expected

    @pytest.mark.parametrize("value", ["float32", "single", "double", "float16", "amp"])
    def test_semantic_precision_aliases_are_rejected(self, value: str) -> None:
        with pytest.raises(ValueError):
            RunSettings.model_validate({"precision": value})


# ============================================================================
# Completeness Validation Tests
# ============================================================================


@pytest.fixture
def _training_job_data(tmp_path: Path) -> dict:
    """Minimal valid TrainingJobConfig dict with data files.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        Dict suitable for TrainingJobConfig.model_validate.
    """
    features_path = tmp_path / "features.npy"
    targets_path = tmp_path / "targets.npy"
    np.save(features_path, np.random.rand(100, 10))
    np.save(targets_path, np.random.rand(100, 1))
    return {
        "run": {"type": "train", "seed": 42},
        "model": {"class": "LinearNetwork", "module_path": "dlkit.domain.nn.ffnn"},
        "data": {
            "class": "FlexibleDataset",
            "batch_size": 16,
            "features": [{"name": "x", "format": "npy", "path": str(features_path)}],
            "targets": [{"name": "y", "format": "npy", "path": str(targets_path)}],
        },
        "training": {
            "trainer": {"max_epochs": 10},
            "optimizer": {"default_optimizer": {"name": "Adam", "lr": 0.001}},
            "loss": {"name": "MSELoss", "module_path": "torch.nn"},
        },
    }


class TestCompletenessValidation:
    """Test completeness validators for build-readiness checks."""

    def test_validate_training_config_complete_with_all_sections_succeeds(
        self, _training_job_data: dict
    ) -> None:
        """Test completeness validation passes when all required sections present."""
        from dlkit.infrastructure.config.job_config import TrainingJobConfig

        config = TrainingJobConfig.model_validate(_training_job_data)

        # Should not raise
        validate_training_config_complete(config)

    def test_validate_training_config_with_empty_data_succeeds(self) -> None:
        """Training config with empty data section passes validate_training_config_complete."""
        from dlkit.infrastructure.config.job_config import TrainingJobConfig

        config_dict = {
            "run": {"type": "train"},
            "model": {"class": "LinearNetwork", "module_path": "dlkit.domain.nn.ffnn"},
            "data": {
                "class": "FlexibleDataset",
            },
            "training": {
                "trainer": {"max_epochs": 10},
                "optimizer": {"default_optimizer": {"name": "Adam", "lr": 0.001}},
                "loss": {"name": "MSELoss", "module_path": "torch.nn"},
            },
        }

        config = TrainingJobConfig.model_validate(config_dict)
        assert config.data is not None
        assert len(config.data.features) == 0
        assert len(config.data.targets) == 0

        # Empty data section is valid — no error
        validate_training_config_complete(config)

    def test_validate_training_config_bad_feature_path_fails(self, tmp_path: Path) -> None:
        """Training config with non-existent feature path fails completeness check."""
        from dlkit.infrastructure.config.job_config import TrainingJobConfig

        targets_path = tmp_path / "targets.npy"
        np.save(targets_path, np.random.rand(100, 1))

        config_dict = {
            "run": {"type": "train"},
            "model": {"class": "LinearNetwork", "module_path": "dlkit.domain.nn.ffnn"},
            "data": {
                "class": "FlexibleDataset",
                "features": [{"name": "x", "format": "npy", "path": "/nonexistent/features.npy"}],
                "targets": [{"name": "y", "format": "npy", "path": str(targets_path)}],
            },
            "training": {
                "trainer": {"max_epochs": 10},
                "optimizer": {"default_optimizer": {"name": "Adam", "lr": 0.001}},
                "loss": {"name": "MSELoss", "module_path": "torch.nn"},
            },
        }

        with pytest.raises(ValidationError):
            TrainingJobConfig.model_validate(config_dict)

    def test_validate_inference_config_complete_with_checkpoint_succeeds(
        self, tmp_path: Path
    ) -> None:
        """Test inference completeness validation passes with valid checkpoint."""
        from dlkit.infrastructure.config.job_config import InferenceJobConfig

        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.write_text("fake checkpoint")

        config_dict = {
            "run": {"type": "predict"},
            "model": {
                "class": "LinearNetwork",
                "module_path": "dlkit.domain.nn.ffnn",
                "checkpoint": str(checkpoint_path),
            },
        }

        config = InferenceJobConfig.model_validate(config_dict)

        # Should not raise
        validate_inference_config_complete(config)

    def test_validate_inference_config_missing_checkpoint_fails(self) -> None:
        """Test inference completeness validation fails without checkpoint."""
        from dlkit.infrastructure.config.job_config import InferenceJobConfig

        config_dict = {
            "run": {"type": "predict"},
            "model": {
                "class": "LinearNetwork",
                "module_path": "dlkit.domain.nn.ffnn",
                # Missing checkpoint!
            },
        }

        # InferenceJobConfig itself validates checkpoint is present
        with pytest.raises((ValidationError, ConfigValidationError)) as exc_info:
            config = InferenceJobConfig.model_validate(config_dict)
            validate_inference_config_complete(config)

        error_msg = str(exc_info.value)
        assert "checkpoint" in error_msg.lower()

    def test_validate_inference_config_nonexistent_checkpoint_fails(self) -> None:
        """Test inference completeness validation fails with non-existent checkpoint."""
        from dlkit.infrastructure.config.job_config import InferenceJobConfig

        config_dict = {
            "run": {"type": "predict"},
            "model": {
                "class": "LinearNetwork",
                "module_path": "dlkit.domain.nn.ffnn",
                "checkpoint": "/this/checkpoint/does/not/exist.ckpt",
            },
        }

        config = InferenceJobConfig.model_validate(config_dict)

        # Completeness validation should fail
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_inference_config_complete(config)

        error_msg = str(exc_info.value)
        assert "checkpoint" in error_msg.lower()

    def test_validate_optimization_config_missing_space_fails(self) -> None:
        """Test search job config requires non-empty search space."""
        from dlkit.infrastructure.config.job_config import SearchJobConfig

        config_dict = {
            "run": {"type": "search"},
            "model": {"class": "LinearNetwork", "module_path": "dlkit.domain.nn.ffnn"},
            "data": {
                "class": "FlexibleDataset",
                "features": [],
                "targets": [],
            },
            "training": {
                "trainer": {"max_epochs": 10},
                "optimizer": {"default_optimizer": {"name": "Adam", "lr": 0.001}},
                "loss": {"name": "MSELoss", "module_path": "torch.nn"},
            },
            "search": {
                "n_trials": 10,
                "space": {},  # Empty space!
            },
        }

        # SearchJobConfig requires non-empty space
        with pytest.raises(ValidationError) as exc_info:
            SearchJobConfig.model_validate(config_dict)

        assert "space" in str(exc_info.value).lower()


# ============================================================================
# Programmatic Override Workflow Tests
# ============================================================================


class TestProgrammaticOverrideWorkflow:
    """Test the complete workflow: partial load → inject → validate → build."""

    def test_complete_programmatic_workflow(self, tmp_path: Path) -> None:
        """Test full workflow: load config → inject updated data → validate → ready for build."""
        from dlkit.infrastructure.config.job_config import TrainingJobConfig

        # Step 1: Load config with empty data section
        config_dict = {
            "run": {"type": "train", "seed": 42},
            "model": {"class": "LinearNetwork", "module_path": "dlkit.domain.nn.ffnn"},
            "data": {"class": "FlexibleDataset"},
            "training": {
                "trainer": {"max_epochs": 10},
                "optimizer": {"default_optimizer": {"name": "Adam", "lr": 0.001}},
                "loss": {"name": "MSELoss", "module_path": "torch.nn"},
            },
        }

        config = TrainingJobConfig.model_validate(config_dict)
        assert config.data is not None
        assert len(config.data.features) == 0

        # Step 2: Create real data files and inject data section
        features_path = tmp_path / "features.npy"
        targets_path = tmp_path / "targets.npy"
        np.save(features_path, np.random.rand(100, 10))
        np.save(targets_path, np.random.rand(100, 1))

        data = DataSettings.model_validate(
            {
                "class": "FlexibleDataset",
                "batch_size": 16,
                "features": [{"name": "x", "format": "npy", "path": str(features_path)}],
                "targets": [{"name": "y", "format": "npy", "path": str(targets_path)}],
            }
        )

        config = config.model_copy(update={"data": data})

        # Step 3: Validate completeness
        validate_training_config_complete(config)  # Should not raise

        # Step 4: Config is now ready for BuildFactory
        assert config.data is not None
        assert len(config.data.features) == 1
        assert len(config.data.targets) == 1

    def test_programmatic_injection_validates_eagerly(self) -> None:
        """Test that programmatic injection validates data eagerly."""
        # Try to inject data with invalid path
        # This should fail eagerly during DataSettings validation
        with pytest.raises(ValidationError):
            DataSettings.model_validate(
                {
                    "class": "FlexibleDataset",
                    "features": [{"name": "x", "format": "npy", "path": "/nonexistent/bad.npy"}],
                    "targets": [{"name": "y", "format": "npy", "path": "/another/bad.npy"}],
                }
            )


# ============================================================================
# Edge Cases and Error Messages
# ============================================================================


class TestEdgeCasesAndErrorMessages:
    """Test edge cases and verify error message quality."""

    def test_inference_job_config_missing_checkpoint_raises(self) -> None:
        """InferenceJobConfig.model_validate raises when checkpoint is missing."""
        from dlkit.infrastructure.config.job_config import InferenceJobConfig

        config_dict = {
            "run": {"type": "predict"},
            "model": {
                "class": "LinearNetwork",
                "module_path": "dlkit.domain.nn.ffnn",
                # No checkpoint
            },
        }

        with pytest.raises((ValidationError, ConfigValidationError)) as exc_info:
            config = InferenceJobConfig.model_validate(config_dict)
            validate_inference_config_complete(config)

        assert "checkpoint" in str(exc_info.value).lower()

    def test_search_job_config_empty_space_fails(self) -> None:
        """SearchJobConfig requires non-empty search space."""
        from dlkit.infrastructure.config.job_config import SearchJobConfig

        config_dict = {
            "run": {"type": "search"},
            "model": {"class": "LinearNetwork", "module_path": "dlkit.domain.nn.ffnn"},
            "data": {"class": "FlexibleDataset"},
            "training": {
                "trainer": {"max_epochs": 1},
                "loss": {"name": "MSELoss", "module_path": "torch.nn"},
            },
            "search": {"n_trials": 5, "space": {}},
        }

        with pytest.raises(ValidationError):
            SearchJobConfig.model_validate(config_dict)

    def test_data_settings_validates_paths_eagerly(self, tmp_path: Path) -> None:
        """DataSettings validates feature/target paths at construction time."""
        features_path = tmp_path / "features.npy"
        targets_path = tmp_path / "targets.npy"
        np.save(features_path, np.random.rand(100, 10))
        np.save(targets_path, np.random.rand(100, 1))

        # Valid config should succeed
        data = DataSettings.model_validate(
            {
                "class": "FlexibleDataset",
                "features": [{"name": "x", "format": "npy", "path": str(features_path)}],
                "targets": [{"name": "y", "format": "npy", "path": str(targets_path)}],
            }
        )
        assert len(data.features) == 1
        assert len(data.targets) == 1

        # Invalid path should fail at construction
        with pytest.raises(ValidationError):
            DataSettings.model_validate(
                {
                    "class": "FlexibleDataset",
                    "features": [{"name": "x", "format": "npy", "path": "/does/not/exist.npy"}],
                }
            )
