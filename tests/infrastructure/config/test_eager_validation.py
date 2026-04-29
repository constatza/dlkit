"""Tests for eager validation architecture with workflow configs.

This module tests the new eager validation system where:
1. Configs validate eagerly at load time (fail-fast on typos/types)
2. Optional sections can be None (programmatic injection supported)
3. Completeness validators ensure configs are build-ready
4. model_copy(update={...}) allows section injection with validation
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from dlkit.infrastructure.config.data_entries import Feature, Target
from dlkit.infrastructure.config.dataloader_settings import DataloaderSettings
from dlkit.infrastructure.config.datamodule_settings import DataModuleSettings
from dlkit.infrastructure.config.dataset_settings import DatasetSettings
from dlkit.infrastructure.config.model_components import (
    LossComponentSettings,
    MetricComponentSettings,
    ModelComponentSettings,
)
from dlkit.infrastructure.config.optimizer_settings import OptimizerSettings
from dlkit.infrastructure.config.optuna_settings import OptunaSettings
from dlkit.infrastructure.config.session_settings import SessionSettings
from dlkit.infrastructure.config.validators import (
    ConfigValidationError,
    validate_inference_config_complete,
    validate_optimization_config_complete,
    validate_training_config_complete,
)
from dlkit.infrastructure.config.workflow_configs import (
    InferenceWorkflowConfig,
    OptimizationWorkflowConfig,
    TrainingWorkflowConfig,
)

# ============================================================================
# Success Cases: Valid Configs Load Successfully
# ============================================================================


class TestEagerValidationSuccessCases:
    """Test that valid configs load successfully with eager validation."""

    def test_training_config_with_all_required_sections_succeeds(self, tmp_path: Path):
        """Test training config with all required sections loads successfully."""
        # Create real data files
        features_path = tmp_path / "features.npy"
        targets_path = tmp_path / "targets.npy"
        np.save(features_path, np.random.rand(100, 10))
        np.save(targets_path, np.random.rand(100, 1))

        config_dict = {
            "SESSION": {
                "name": "test_training",
                "seed": 42,
                "root_dir": str(tmp_path),
            },
            "TRAINING": {
                "epochs": 10,
                "optimizer": {"name": "Adam", "lr": 0.001},
                "loss_function": {"name": "MSELoss", "module_path": "torch.nn"},
            },
            "DATAMODULE": {
                "name": "InMemoryModule",
                "module_path": "dlkit.engine.adapters.lightning.datamodules",
                "dataloader": {"batch_size": 16},
            },
            "DATASET": {
                "features": [
                    {"name": "x", "path": str(features_path)},
                ],
                "targets": [
                    {"name": "y", "path": str(targets_path)},
                ],
            },
            "MODEL": {
                "name": "LinearNetwork",
                "module_path": "dlkit.domain.nn.ffnn",
                "input_size": 10,
                "output_size": 1,
            },
        }

        # Should load successfully with eager validation
        config = TrainingWorkflowConfig.model_validate(config_dict)

        # Verify all sections present
        assert config.SESSION is not None
        assert config.TRAINING is not None
        assert config.DATAMODULE is not None
        assert config.DATASET is not None
        assert config.MODEL is not None

        # Verify values correct
        assert config.SESSION.name == "test_training"
        assert config.TRAINING.epochs == 10
        assert config.DATAMODULE.dataloader.batch_size == 16

    def test_training_config_optional_sections_can_be_omitted(self):
        """Test that optional sections (MLFLOW, OPTUNA) can be omitted."""
        config_dict = {
            "SESSION": {
                "name": "minimal_training",
                "seed": 42,
            },
            "TRAINING": {
                "epochs": 5,
                "optimizer": {"name": "Adam", "lr": 0.001},
                "loss_function": {"name": "MSELoss", "module_path": "torch.nn"},
            },
        }

        # Should load successfully without DATAMODULE/DATASET/MODEL
        config = TrainingWorkflowConfig.model_validate(config_dict)

        # Required sections present
        assert config.SESSION is not None
        assert config.TRAINING is not None

        # Optional sections are None
        assert config.DATAMODULE is None
        assert config.DATASET is None
        assert config.MODEL is None

        # Default values for tracking sections
        assert config.MLFLOW is None
        assert config.OPTUNA.enabled is False

    def test_inference_config_with_checkpoint_succeeds(self, tmp_path: Path):
        """Test inference config with valid checkpoint loads successfully."""
        # Create dummy checkpoint
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.write_text("fake checkpoint")

        config_dict = {
            "SESSION": {
                "name": "test_inference",
                "workflow": "inference",
                "seed": 123,
            },
            "MODEL": {
                "name": "LinearNetwork",
                "module_path": "dlkit.domain.nn.ffnn",
                "checkpoint": str(checkpoint_path),
            },
        }

        # Should load successfully
        config = InferenceWorkflowConfig.model_validate(config_dict)

        assert config.SESSION.is_inference_mode is True
        # Checkpoint can be str or Path - just verify it's set
        assert config.MODEL.checkpoint is not None
        assert str(config.MODEL.checkpoint) == str(checkpoint_path)

    def test_inference_config_exposes_has_dataset_config(self, tmp_path: Path):
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.write_text("fake checkpoint")

        config = InferenceWorkflowConfig.model_validate(
            {
                "SESSION": {"name": "predict", "workflow": "inference"},
                "MODEL": {
                    "name": "LinearNetwork",
                    "module_path": "dlkit.domain.nn.ffnn",
                    "checkpoint": str(checkpoint_path),
                },
                "DATAMODULE": {
                    "name": "InMemoryModule",
                    "module_path": "dlkit.engine.adapters.lightning.datamodules",
                },
                "DATASET": {"name": "FlexibleDataset"},
            }
        )

        assert config.has_dataset_config is True

    def test_programmatic_section_injection_succeeds(self, tmp_path: Path):
        """Test that sections can be injected programmatically after load."""
        # Create real data files
        features_path = tmp_path / "features.npy"
        targets_path = tmp_path / "targets.npy"
        np.save(features_path, np.random.rand(100, 10))
        np.save(targets_path, np.random.rand(100, 1))

        # Load partial config
        config_dict = {
            "SESSION": {"name": "test_injection", "seed": 42},
            "TRAINING": {
                "epochs": 10,
                "optimizer": {"name": "Adam", "lr": 0.001},
                "loss_function": {"name": "MSELoss", "module_path": "torch.nn"},
            },
        }

        config = TrainingWorkflowConfig.model_validate(config_dict)
        assert config.DATASET is None

        # Inject DATASET section programmatically (with eager validation)
        dataset = DatasetSettings(
            features=(Feature(name="x", path=features_path),),
            targets=(Target(name="y", path=targets_path),),
        )

        config = config.model_copy(update={"DATASET": dataset})

        # Verify injection succeeded
        assert config.DATASET is not None
        assert len(config.DATASET.features) == 1
        assert len(config.DATASET.targets) == 1


# ============================================================================
# Failure Cases: Invalid Configs Fail at Load Time
# ============================================================================


class TestEagerValidationFailureCases:
    """Test that invalid configs fail immediately at load time."""

    def test_invalid_path_fails_immediately_at_load_time(self):
        """Test that configs with invalid paths fail at load (not at build time)."""
        config_dict = {
            "SESSION": {"name": "test_bad_path", "seed": 42},
            "TRAINING": {
                "epochs": 10,
                "optimizer": {"name": "Adam", "lr": 0.001},
                "loss_function": {"name": "MSELoss", "module_path": "torch.nn"},
            },
            "DATASET": {
                "features": [
                    {"name": "x", "path": "/this/path/does/not/exist.npy"},
                ],
            },
        }

        # Should fail at load time with ValidationError
        with pytest.raises(ValidationError) as exc_info:
            TrainingWorkflowConfig.model_validate(config_dict)

        # Error should mention the path issue
        error_msg = str(exc_info.value)
        assert "path" in error_msg.lower() or "exist" in error_msg.lower()

    def test_type_error_fails_immediately(self):
        """Test that type errors fail immediately at load time."""
        config_dict = {
            "SESSION": {"name": "test_type_error", "seed": 42},
            "TRAINING": {
                "epochs": "not_an_integer",  # Wrong type!
                "optimizer": {"name": "Adam", "lr": 0.001},
                "loss_function": {"name": "MSELoss", "module_path": "torch.nn"},
            },
        }

        # Should fail with validation error
        with pytest.raises(ValidationError):
            TrainingWorkflowConfig.model_validate(config_dict)

    def test_missing_required_section_fails(self):
        """Test that missing required sections (SESSION, TRAINING) fail at load."""
        # Missing SESSION
        config_dict_no_session = {
            "TRAINING": {
                "epochs": 10,
                "optimizer": {"name": "Adam", "lr": 0.001},
                "loss_function": {"name": "MSELoss", "module_path": "torch.nn"},
            },
        }

        with pytest.raises(ValidationError) as exc_info:
            TrainingWorkflowConfig.model_validate(config_dict_no_session)

        error_msg = str(exc_info.value)
        assert "SESSION" in error_msg

        # Missing TRAINING
        config_dict_no_training = {
            "SESSION": {"name": "test", "seed": 42},
        }

        with pytest.raises(ValidationError) as exc_info:
            TrainingWorkflowConfig.model_validate(config_dict_no_training)

        error_msg = str(exc_info.value)
        assert "TRAINING" in error_msg

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: DataModuleSettings(module_path="dlkit.not_a_real_module"),
            lambda: DatasetSettings(
                name="BrokenDataset",
                module_path="dlkit.not_a_real_module",
            ),
            lambda: ModelComponentSettings(
                name="BrokenModel",
                module_path="dlkit.not_a_real_module",
            ),
            lambda: OptimizerSettings(module_path="dlkit.not_a_real_module"),
            lambda: LossComponentSettings(module_path="dlkit.not_a_real_module"),
            lambda: MetricComponentSettings(module_path="dlkit.not_a_real_module"),
            lambda: OptunaSettings(sampler={"module_path": "dlkit.not_a_real_module"}),
            lambda: OptunaSettings(pruner={"module_path": "dlkit.not_a_real_module"}),
        ],
    )
    def test_invalid_module_paths_fail_at_load_time(self, factory) -> None:
        with pytest.raises(ValidationError, match="module_path"):
            factory()


class TestWorkflowCrossValidation:
    """Test workflow-level cross-section validation."""

    def test_optimization_config_warns_for_unknown_optuna_model_keys(self) -> None:
        config_dict = {
            "SESSION": {"name": "optuna_warning", "workflow": "optimize"},
            "TRAINING": {"epochs": 2},
            "OPTUNA": {"enabled": True, "model": {"bogus": {"low": 1, "high": 2}}},
            "MODEL": {"name": "LinearNetwork", "module_path": "dlkit.domain.nn.ffnn"},
        }

        with pytest.warns(UserWarning, match="OPTUNA.model contains keys not in MODEL"):
            OptimizationWorkflowConfig.model_validate(config_dict)

    def test_optimization_config_allows_optuna_keys_for_model_extra_fields(self) -> None:
        config_dict = {
            "SESSION": {"name": "optuna_extra", "workflow": "optimize"},
            "TRAINING": {"epochs": 2},
            "OPTUNA": {"enabled": True, "model": {"dropout": {"low": 0.1, "high": 0.5}}},
            "MODEL": {
                "name": "LinearNetwork",
                "module_path": "dlkit.domain.nn.ffnn",
                "dropout": 0.2,
            },
        }

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            OptimizationWorkflowConfig.model_validate(config_dict)

        assert not recorded


class TestSessionPrecisionAliases:
    """Ensure session precision accepts Lightning-style aliases."""

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
        settings = SessionSettings(precision=value)
        assert str(settings.precision) == expected


# ============================================================================
# Completeness Validation Tests
# ============================================================================


class TestCompletenessValidation:
    """Test completeness validators for build-readiness checks."""

    def test_validate_training_config_complete_with_all_sections_succeeds(self, tmp_path: Path):
        """Test completeness validation passes when all required sections present."""
        # Create real data files
        features_path = tmp_path / "features.npy"
        targets_path = tmp_path / "targets.npy"
        np.save(features_path, np.random.rand(100, 10))
        np.save(targets_path, np.random.rand(100, 1))

        config_dict = {
            "SESSION": {"name": "test_complete", "seed": 42},
            "TRAINING": {
                "epochs": 10,
                "optimizer": {"name": "Adam", "lr": 0.001},
                "loss_function": {"name": "MSELoss", "module_path": "torch.nn"},
            },
            "DATAMODULE": {
                "name": "InMemoryModule",
                "module_path": "dlkit.engine.adapters.lightning.datamodules",
                "dataloader": {"batch_size": 16},
            },
            "DATASET": {
                "features": [{"name": "x", "path": str(features_path)}],
                "targets": [{"name": "y", "path": str(targets_path)}],
            },
            "MODEL": {
                "name": "LinearNetwork",
                "module_path": "dlkit.domain.nn.ffnn",
            },
        }

        config = TrainingWorkflowConfig.model_validate(config_dict)

        # Should not raise
        validate_training_config_complete(config)

    def test_validate_training_config_missing_datamodule_fails(self):
        """Test completeness validation fails when DATAMODULE missing."""
        config_dict = {
            "SESSION": {"name": "test_incomplete", "seed": 42},
            "TRAINING": {
                "epochs": 10,
                "optimizer": {"name": "Adam", "lr": 0.001},
                "loss_function": {"name": "MSELoss", "module_path": "torch.nn"},
            },
            "MODEL": {
                "name": "LinearNetwork",
                "module_path": "dlkit.domain.nn.ffnn",
            },
        }

        config = TrainingWorkflowConfig.model_validate(config_dict)
        assert config.DATAMODULE is None

        # Completeness validation should fail
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_training_config_complete(config)

        error_msg = str(exc_info.value)
        assert "DATAMODULE" in error_msg
        assert "required" in error_msg.lower()

    def test_validate_training_config_missing_dataset_fails(self):
        """Test completeness validation fails when DATASET missing."""
        config_dict = {
            "SESSION": {"name": "test_incomplete", "seed": 42},
            "TRAINING": {
                "epochs": 10,
                "optimizer": {"name": "Adam", "lr": 0.001},
                "loss_function": {"name": "MSELoss", "module_path": "torch.nn"},
            },
            "DATAMODULE": {
                "name": "InMemoryModule",
                "module_path": "dlkit.engine.adapters.lightning.datamodules",
                "dataloader": {"batch_size": 16},
            },
            "MODEL": {
                "name": "LinearNetwork",
                "module_path": "dlkit.domain.nn.ffnn",
            },
        }

        config = TrainingWorkflowConfig.model_validate(config_dict)
        assert config.DATASET is None

        # Completeness validation should fail
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_training_config_complete(config)

        error_msg = str(exc_info.value)
        assert "DATASET" in error_msg

    def test_validate_training_config_missing_model_fails(self, tmp_path: Path):
        """Test completeness validation fails when MODEL missing."""
        # Create real data files
        features_path = tmp_path / "features.npy"
        targets_path = tmp_path / "targets.npy"
        np.save(features_path, np.random.rand(100, 10))
        np.save(targets_path, np.random.rand(100, 1))

        config_dict = {
            "SESSION": {"name": "test_incomplete", "seed": 42},
            "TRAINING": {
                "epochs": 10,
                "optimizer": {"name": "Adam", "lr": 0.001},
                "loss_function": {"name": "MSELoss", "module_path": "torch.nn"},
            },
            "DATAMODULE": {
                "name": "InMemoryModule",
                "module_path": "dlkit.engine.adapters.lightning.datamodules",
                "dataloader": {"batch_size": 16},
            },
            "DATASET": {
                "features": [{"name": "x", "path": str(features_path)}],
                "targets": [{"name": "y", "path": str(targets_path)}],
            },
        }

        config = TrainingWorkflowConfig.model_validate(config_dict)
        assert config.MODEL is None

        # Completeness validation should fail
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_training_config_complete(config)

        error_msg = str(exc_info.value)
        assert "MODEL" in error_msg

    def test_validate_inference_config_complete_with_checkpoint_succeeds(self, tmp_path: Path):
        """Test inference completeness validation passes with valid checkpoint."""
        # Create dummy checkpoint
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.write_text("fake checkpoint")

        config_dict = {
            "SESSION": {
                "name": "test_inference",
                "workflow": "inference",
                "seed": 123,
            },
            "MODEL": {
                "name": "LinearNetwork",
                "module_path": "dlkit.domain.nn.ffnn",
                "checkpoint": str(checkpoint_path),
            },
        }

        config = InferenceWorkflowConfig.model_validate(config_dict)

        # Should not raise
        validate_inference_config_complete(config)

    def test_validate_inference_config_missing_checkpoint_fails(self):
        """Test inference completeness validation fails without checkpoint."""
        config_dict = {
            "SESSION": {
                "name": "test_inference",
                "workflow": "inference",
                "seed": 123,
            },
            "MODEL": {
                "name": "LinearNetwork",
                "module_path": "dlkit.domain.nn.ffnn",
                # Missing checkpoint!
            },
        }

        config = InferenceWorkflowConfig.model_validate(config_dict)

        # Completeness validation should fail
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_inference_config_complete(config)

        error_msg = str(exc_info.value)
        assert "checkpoint" in error_msg.lower()

    def test_validate_inference_config_nonexistent_checkpoint_fails(self):
        """Test inference completeness validation fails with non-existent checkpoint."""
        config_dict = {
            "SESSION": {
                "name": "test_inference",
                "workflow": "inference",
                "seed": 123,
            },
            "MODEL": {
                "name": "LinearNetwork",
                "module_path": "dlkit.domain.nn.ffnn",
                "checkpoint": "/this/checkpoint/does/not/exist.ckpt",
            },
        }

        config = InferenceWorkflowConfig.model_validate(config_dict)

        # Completeness validation should fail
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_inference_config_complete(config)

        error_msg = str(exc_info.value)
        assert "checkpoint" in error_msg.lower()
        assert "exist" in error_msg.lower()

    def test_validate_optimization_config_complete_succeeds(self, tmp_path: Path):
        """Test optimization completeness validation passes with all sections."""
        # Create real data files
        features_path = tmp_path / "features.npy"
        targets_path = tmp_path / "targets.npy"
        np.save(features_path, np.random.rand(100, 10))
        np.save(targets_path, np.random.rand(100, 1))

        config_dict = {
            "SESSION": {"name": "test_optim", "seed": 42, "workflow": "optimize"},
            "TRAINING": {
                "epochs": 10,
                "optimizer": {"name": "Adam", "lr": 0.001},
                "loss_function": {"name": "MSELoss", "module_path": "torch.nn"},
            },
            "OPTUNA": {
                "enabled": True,
                "n_trials": 10,
                "model": {"lr": [0.0001, 0.01]},
            },
            "DATAMODULE": {
                "name": "InMemoryModule",
                "module_path": "dlkit.engine.adapters.lightning.datamodules",
                "dataloader": {"batch_size": 16},
            },
            "DATASET": {
                "features": [{"name": "x", "path": str(features_path)}],
                "targets": [{"name": "y", "path": str(targets_path)}],
            },
            "MODEL": {
                "name": "LinearNetwork",
                "module_path": "dlkit.domain.nn.ffnn",
            },
        }

        config = OptimizationWorkflowConfig.model_validate(config_dict)

        # Should not raise
        validate_optimization_config_complete(config)

    def test_validate_optimization_config_optuna_disabled_fails(self):
        """Test optimization completeness validation fails if Optuna disabled."""
        config_dict = {
            "SESSION": {"name": "test_optim", "seed": 42, "workflow": "optimize"},
            "TRAINING": {
                "epochs": 10,
                "optimizer": {"name": "Adam", "lr": 0.001},
                "loss_function": {"name": "MSELoss", "module_path": "torch.nn"},
            },
            "OPTUNA": {
                "enabled": False,  # Disabled!
                "n_trials": 10,
            },
        }

        config = OptimizationWorkflowConfig.model_validate(config_dict)

        # Completeness validation should fail
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_optimization_config_complete(config)

        error_msg = str(exc_info.value)
        assert "enabled" in error_msg.lower()
        assert "OPTUNA" in error_msg


# ============================================================================
# Programmatic Override Workflow Tests
# ============================================================================


class TestProgrammaticOverrideWorkflow:
    """Test the complete workflow: partial load → inject → validate → build."""

    def test_complete_programmatic_workflow(self, tmp_path: Path):
        """Test full workflow: load partial → inject DATASET → validate → ready for build."""
        # Step 1: Load partial config (only required sections)
        config_dict = {
            "SESSION": {"name": "test_workflow", "seed": 42, "root_dir": str(tmp_path)},
            "TRAINING": {
                "epochs": 10,
                "optimizer": {"name": "Adam", "lr": 0.001},
                "loss_function": {"name": "MSELoss", "module_path": "torch.nn"},
            },
        }

        config = TrainingWorkflowConfig.model_validate(config_dict)
        assert config.DATASET is None
        assert config.DATAMODULE is None
        assert config.MODEL is None

        # Step 2: Inject sections programmatically
        # Create real data files
        features_path = tmp_path / "features.npy"
        targets_path = tmp_path / "targets.npy"
        np.save(features_path, np.random.rand(100, 10))
        np.save(targets_path, np.random.rand(100, 1))

        dataset = DatasetSettings(
            features=(Feature(name="x", path=features_path),),
            targets=(Target(name="y", path=targets_path),),
        )

        datamodule = DataModuleSettings(
            name="InMemoryModule",
            module_path="dlkit.engine.adapters.lightning.datamodules",
            dataloader=DataloaderSettings(batch_size=16),
        )

        model = ModelComponentSettings.model_validate(
            {
                "name": "LinearNetwork",
                "module_path": "dlkit.domain.nn.ffnn",
                "input_size": 10,
                "output_size": 1,
            }
        )

        config = config.model_copy(
            update={
                "DATASET": dataset,
                "DATAMODULE": datamodule,
                "MODEL": model,
            }
        )

        # Step 3: Validate completeness
        validate_training_config_complete(config)  # Should not raise

        # Step 4: Config is now ready for BuildFactory
        assert config.DATASET is not None
        assert config.DATAMODULE is not None
        assert config.MODEL is not None

    def test_programmatic_injection_validates_eagerly(self):
        """Test that programmatic injection validates data eagerly."""
        # Load partial config
        config_dict = {
            "SESSION": {"name": "test_validation", "seed": 42},
            "TRAINING": {
                "epochs": 10,
                "optimizer": {"name": "Adam", "lr": 0.001},
                "loss_function": {"name": "MSELoss", "module_path": "torch.nn"},
            },
        }

        config = TrainingWorkflowConfig.model_validate(config_dict)

        # Try to inject DATASET with invalid path
        # This should fail eagerly during model_copy due to Pydantic validation
        with pytest.raises(ValidationError):
            dataset = DatasetSettings(
                features=(Feature(name="x", path="/nonexistent/bad.npy"),),
                targets=(Target(name="y", path="/another/bad.npy"),),
            )

            config.model_copy(update={"DATASET": dataset})


# ============================================================================
# Edge Cases and Error Messages
# ============================================================================


class TestEdgeCasesAndErrorMessages:
    """Test edge cases and verify error message quality."""

    def test_empty_dataset_features_and_targets_fails_completeness(self):
        """Test that DATASET with no features or targets fails completeness check."""
        config_dict = {
            "SESSION": {"name": "test_empty", "seed": 42},
            "TRAINING": {
                "epochs": 10,
                "optimizer": {"name": "Adam", "lr": 0.001},
                "loss_function": {"name": "MSELoss", "module_path": "torch.nn"},
            },
            "DATAMODULE": {
                "name": "InMemoryModule",
                "module_path": "dlkit.engine.adapters.lightning.datamodules",
                "dataloader": {"batch_size": 16},
            },
            "DATASET": {
                # Empty - no features or targets
            },
            "MODEL": {
                "name": "LinearNetwork",
                "module_path": "dlkit.domain.nn.ffnn",
            },
        }

        config = TrainingWorkflowConfig.model_validate(config_dict)

        # Completeness validation should fail
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_training_config_complete(config)

        error_msg = str(exc_info.value)
        assert "feature" in error_msg.lower() or "target" in error_msg.lower()

    def test_error_messages_are_actionable(self):
        """Test that error messages provide clear guidance."""
        config_dict = {
            "SESSION": {"name": "test_errors", "seed": 42},
            "TRAINING": {
                "epochs": 10,
                "optimizer": {"name": "Adam", "lr": 0.001},
                "loss_function": {"name": "MSELoss", "module_path": "torch.nn"},
            },
        }

        config = TrainingWorkflowConfig.model_validate(config_dict)

        # Missing sections error should be clear
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_training_config_complete(config)

        error_msg = str(exc_info.value)
        # Error should list missing sections
        assert "DATAMODULE" in error_msg
        assert "DATASET" in error_msg
        assert "MODEL" in error_msg
        # Error should provide guidance
        assert "required" in error_msg.lower()
        assert "TOML" in error_msg or "inject" in error_msg.lower()

    def test_convenience_properties_work_correctly(self, tmp_path: Path):
        """Test that convenience properties on config objects work correctly."""
        features_path = tmp_path / "features.npy"
        targets_path = tmp_path / "targets.npy"
        np.save(features_path, np.random.rand(100, 10))
        np.save(targets_path, np.random.rand(100, 1))

        config_dict = {
            "SESSION": {"name": "test_props", "seed": 42},
            "TRAINING": {
                "epochs": 10,
                "optimizer": {"name": "Adam", "lr": 0.001},
                "loss_function": {"name": "MSELoss", "module_path": "torch.nn"},
            },
            "DATAMODULE": {
                "name": "InMemoryModule",
                "module_path": "dlkit.engine.adapters.lightning.datamodules",
                "dataloader": {"batch_size": 16},
            },
            "DATASET": {
                "features": [{"name": "x", "path": str(features_path)}],
                "targets": [{"name": "y", "path": str(targets_path)}],
            },
            "MODEL": {
                "name": "LinearNetwork",
                "module_path": "dlkit.domain.nn.ffnn",
            },
            "MLFLOW": {},
            "OPTUNA": {
                "enabled": False,
            },
        }

        config = TrainingWorkflowConfig.model_validate(config_dict)

        # Test convenience properties
        assert config.mlflow_enabled is True
        assert config.optuna_enabled is False
        assert config.has_complete_data_config is True
        assert config.has_model_config is True
