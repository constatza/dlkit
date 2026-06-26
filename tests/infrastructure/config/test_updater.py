"""Tests for settings updater with deep merge functionality."""

from pathlib import Path
from typing import cast

import numpy as np
import pytest
from pydantic import ValidationError

from dlkit.infrastructure.config import load_job, update_settings
from dlkit.infrastructure.config.data_roles import DataRole
from dlkit.infrastructure.config.entry_types import NpyEntry
from dlkit.infrastructure.config.job_config import TrainingJobConfig
from dlkit.infrastructure.config.optimizer_component import (
    AdamSettings,
    AdamWSettings,
    LBFGSSettings,
    MuonSettings,
)
from dlkit.infrastructure.config.training_settings import TrainingSettings


def _expect_not_none[T](value: T | None) -> T:
    assert value is not None
    return value


def _expect_training_job(settings: object) -> TrainingJobConfig:
    assert isinstance(settings, TrainingJobConfig)
    return settings


def _expect_optimizer_with_lr(
    training: TrainingSettings,
) -> AdamWSettings | AdamSettings | LBFGSSettings | MuonSettings:
    optimizer = training.optimizer.default_optimizer
    assert isinstance(optimizer, AdamWSettings | AdamSettings | LBFGSSettings | MuonSettings)
    return optimizer


def _expect_optimizer_with_weight_decay(
    training: TrainingSettings,
) -> AdamWSettings | AdamSettings:
    optimizer = training.optimizer.default_optimizer
    assert isinstance(optimizer, AdamWSettings | AdamSettings)
    return optimizer


@pytest.fixture
def minimal_train_toml() -> str:
    """Minimal training job TOML with required sections."""
    return """
[run]
type = "train"

[experiment]
name = "test-session"

[model]
class = "LinearNetwork"
module_path = "dlkit.domain.nn.ffnn"

[data]
batch_size = 32

[training]
loss = "mse"

[training.optimizer.default_optimizer]
name = "Adam"
lr = 0.001

[training.trainer]
max_epochs = 10
accelerator = "cpu"
"""


@pytest.fixture
def train_config_file(tmp_path: Path, minimal_train_toml: str) -> Path:
    """Write minimal training TOML to a temp file."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(minimal_train_toml)
    return config_path


class TestBasicUpdates:
    """Test basic update functionality."""

    def test_simple_top_level_update(self, train_config_file: Path) -> None:
        """Test updating a simple top-level field."""
        settings = load_job(train_config_file)
        job = _expect_training_job(settings)

        new_settings = update_settings(job, {"experiment": {"name": "new_name"}})

        assert _expect_not_none(new_settings.experiment).name == "new_name"
        # Original unchanged
        assert _expect_not_none(job.experiment).name == "test-session"

    def test_deep_nested_update(self, train_config_file: Path) -> None:
        """Test updating deeply nested fields (3 levels)."""
        settings = load_job(train_config_file)
        job = _expect_training_job(settings)

        new_settings = update_settings(job, {"training": {"trainer": {"max_epochs": 200}}})

        assert new_settings.training.trainer.max_epochs == 200
        # Optimizer preserved
        assert (
            _expect_optimizer_with_lr(new_settings.training).lr
            == _expect_optimizer_with_lr(job.training).lr
        )

    def test_multiple_sections_at_once(self, train_config_file: Path) -> None:
        """Test updating multiple top-level sections simultaneously."""
        settings = load_job(train_config_file)
        job = _expect_training_job(settings)

        new_settings = update_settings(
            job,
            {
                "experiment": {"name": "updated_session"},
                "training": {"trainer": {"max_epochs": 100}},
                "data": {"batch_size": 64},
            },
        )

        assert _expect_not_none(new_settings.experiment).name == "updated_session"
        assert new_settings.training.trainer.max_epochs == 100
        assert new_settings.data.batch_size == 64

        # Unspecified fields preserved
        assert (
            _expect_optimizer_with_lr(new_settings.training).lr
            == _expect_optimizer_with_lr(job.training).lr
        )


class TestPartialUpdatesPreservation:
    """Test that partial updates preserve unspecified fields."""

    @pytest.fixture
    def config_with_weight_decay(self, tmp_path: Path) -> Path:
        """Training config with optimizer weight decay."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[run]
type = "train"

[experiment]
name = "test"

[model]
class = "LinearNetwork"
module_path = "dlkit.domain.nn.ffnn"

[data]
batch_size = 32

[training]
loss = "mse"

[training.optimizer.default_optimizer]
name = "Adam"
lr = 0.001
weight_decay = 0.01

[training.trainer]
max_epochs = 10
accelerator = "cpu"
""")
        return config_path

    def test_partial_update_preserves_sibling_fields(self, config_with_weight_decay: Path) -> None:
        """Test that updating one field preserves sibling fields."""
        settings = load_job(config_with_weight_decay)
        job = _expect_training_job(settings)

        new_settings = update_settings(
            job, {"training": {"optimizer": {"default_optimizer": {"lr": 0.01}}}}
        )

        assert _expect_optimizer_with_lr(new_settings.training).lr == 0.01
        assert _expect_optimizer_with_weight_decay(new_settings.training).weight_decay == 0.01
        assert (
            new_settings.training.optimizer.default_optimizer.name
            == job.training.optimizer.default_optimizer.name
        )


class TestNonDictTypes:
    """Test handling of non-dict types (list, tuple, str, int, etc.)."""

    def test_list_overwrite(self, tmp_path: Path) -> None:
        """Test that lists are overwritten, not merged."""
        features_path = tmp_path / "features.npy"
        np.save(features_path, np.random.rand(100, 10))

        config_path = tmp_path / "config.toml"
        config_path.write_text(f"""
[run]
type = "train"

[experiment]
name = "test"

[model]
class = "LinearNetwork"
module_path = "dlkit.domain.nn.ffnn"

[data]
batch_size = 32

[[data.features]]
name = "old_feature"
format = "npy"
path = "{features_path.as_posix()}"

[training]
loss = "mse"

[training.optimizer.default_optimizer]
name = "Adam"
lr = 0.001

[training.trainer]
max_epochs = 10
accelerator = "cpu"
""")

        settings = load_job(config_path)
        job = _expect_training_job(settings)

        new_settings = update_settings(
            job,
            {
                "data": {
                    "features": (
                        NpyEntry(
                            name="new_feature1",
                            path=features_path,
                            data_role=DataRole.FEATURE,
                        ),
                        NpyEntry(
                            name="new_feature2",
                            path=features_path,
                            data_role=DataRole.FEATURE,
                        ),
                    )
                }
            },
        )

        assert len(new_settings.data.features) == 2
        assert new_settings.data.features[0].name == "new_feature1"
        assert new_settings.data.features[1].name == "new_feature2"

    def test_string_overwrite(self, train_config_file: Path) -> None:
        """Test that strings are overwritten."""
        settings = load_job(train_config_file)
        job = _expect_training_job(settings)

        new_settings = update_settings(job, {"experiment": {"name": "updated"}})

        assert _expect_not_none(new_settings.experiment).name == "updated"

    def test_int_overwrite(self, train_config_file: Path) -> None:
        """Test that integers are overwritten."""
        settings = load_job(train_config_file)
        job = _expect_training_job(settings)

        new_settings = update_settings(job, {"training": {"trainer": {"max_epochs": 999}}})

        assert new_settings.training.trainer.max_epochs == 999

    def test_path_overwrite(self, tmp_path: Path) -> None:
        """Test that Path objects are overwritten."""
        ckpt1 = tmp_path / "model1.ckpt"
        ckpt2 = tmp_path / "model2.ckpt"
        ckpt1.touch()
        ckpt2.touch()

        config_path = tmp_path / "config.toml"
        config_path.write_text(f"""
[run]
type = "train"

[experiment]
name = "test"

[model]
class = "LinearNetwork"
module_path = "dlkit.domain.nn.ffnn"
checkpoint = "{ckpt1.as_posix()}"

[data]
batch_size = 32

[training]
loss = "mse"

[training.optimizer.default_optimizer]
name = "Adam"
lr = 0.001

[training.trainer]
max_epochs = 10
accelerator = "cpu"
""")

        settings = load_job(config_path)
        job = _expect_training_job(settings)

        new_settings = update_settings(job, {"model": {"checkpoint": ckpt2}})

        assert _expect_not_none(new_settings.model).checkpoint == ckpt2


class TestPydanticModelUpdates:
    """Ensure BaseModel updates don't reset existing config sections to defaults."""

    @pytest.fixture
    def config_with_adam(self, tmp_path: Path) -> Path:
        """Config with Adam optimizer and weight_decay."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[run]
type = "train"

[experiment]
name = "test"

[model]
class = "LinearNetwork"
module_path = "dlkit.domain.nn.ffnn"

[data]
batch_size = 32

[training]
loss = "mse"

[training.optimizer.default_optimizer]
name = "Adam"
lr = 0.005
weight_decay = 0.1

[training.trainer]
max_epochs = 10
accelerator = "cpu"
""")
        return config_path

    @pytest.fixture
    def config_with_adamw(self, tmp_path: Path) -> Path:
        """Config with AdamW optimizer."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[run]
type = "train"

[experiment]
name = "test"

[model]
class = "LinearNetwork"
module_path = "dlkit.domain.nn.ffnn"

[data]
batch_size = 32

[training]
loss = "mse"

[training.optimizer.default_optimizer]
name = "AdamW"
lr = 0.01
weight_decay = 0.2

[training.trainer]
max_epochs = 10
accelerator = "cpu"
""")
        return config_path

    def test_training_model_update_preserves_nested_values(self, config_with_adam: Path) -> None:
        """Applying training patch keeps optimizer overrides from config."""
        settings = load_job(config_with_adam)
        job = _expect_training_job(settings)

        assert job.training.optimizer.default_optimizer.name == "Adam"
        assert _expect_optimizer_with_lr(job.training).lr == 0.005
        assert _expect_optimizer_with_weight_decay(job.training).weight_decay == 0.1

        updated = update_settings(job, {"training": {"trainer": {"max_epochs": 50}}})

        assert updated.training.optimizer.default_optimizer.name == "Adam"
        assert _expect_optimizer_with_lr(updated.training).lr == pytest.approx(0.005)
        assert _expect_optimizer_with_weight_decay(updated.training).weight_decay == pytest.approx(
            0.1
        )

    def test_optimizer_model_update_preserves_existing_fields(
        self, config_with_adamw: Path
    ) -> None:
        """Applying optimizer policy patch merges into existing optimizer config."""
        settings = load_job(config_with_adamw)
        job = _expect_training_job(settings)

        assert job.training.optimizer.default_optimizer.name == "AdamW"
        assert _expect_optimizer_with_weight_decay(job.training).weight_decay == 0.2

        updated = update_settings(
            job,
            {"training": {"optimizer": {"default_optimizer": {"lr": 0.123}}}},
        )

        assert updated.training.optimizer.default_optimizer.name == "AdamW"
        assert _expect_optimizer_with_weight_decay(updated.training).weight_decay == pytest.approx(
            0.2
        )
        assert _expect_optimizer_with_lr(updated.training).lr == pytest.approx(0.123)


class TestValidation:
    """Test validation behavior."""

    def test_validation_catches_bad_paths(self, train_config_file: Path) -> None:
        """Test that validation catches invalid file paths."""
        settings = load_job(train_config_file)
        job = _expect_training_job(settings)

        with pytest.raises(ValidationError):
            update_settings(
                job,
                {
                    "data": {
                        "features": (
                            NpyEntry(
                                name="x",
                                path=cast("Path | None", "/nonexistent/bad/path.npy"),
                                data_role=DataRole.FEATURE,
                            ),
                        )
                    }
                },
            )

    def test_validation_with_default_catches_errors(self, train_config_file: Path) -> None:
        """Test that validation is enabled by default and catches errors."""
        settings = load_job(train_config_file)
        job = _expect_training_job(settings)

        new_settings = update_settings(job, {"experiment": {"name": "updated"}})

        assert _expect_not_none(new_settings.experiment).name == "updated"


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_dict(self, train_config_file: Path) -> None:
        """Test updating with empty dict."""
        settings = load_job(train_config_file)
        job = _expect_training_job(settings)

        new_settings = update_settings(job, {})

        assert new_settings.model_dump() == job.model_dump()


class TestInstanceHelper:
    """Test the BasicSettings.update_with convenience method."""

    def test_update_with_matches_functional_helper(self, train_config_file: Path) -> None:
        """Ensure update_with delegates to update_settings and preserves type."""
        settings = load_job(train_config_file)
        job = _expect_training_job(settings)

        via_method = job.update_with({"experiment": {"name": "updated_name"}})

        assert _expect_not_none(via_method.experiment).name == "updated_name"
        # update_with returns a NEW instance (immutable semantics)
        assert via_method is not job
        # Original is unchanged
        assert _expect_not_none(job.experiment).name == "test-session"
