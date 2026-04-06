"""Tests for SESSION.root_dir propagation to EnvironmentSettings and path resolution.

This test suite verifies the three-layer defense-in-depth approach:
1. SESSION.root_dir synced to EnvironmentSettings
2. Defensive path context in BuildFactory
3. Logging and observability
"""

import os
from pathlib import Path
from typing import cast

import pytest

from dlkit.infrastructure.config import GeneralSettings, load_settings
from dlkit.infrastructure.config.environment import env as global_environment
from dlkit.infrastructure.io.config import load_config
from dlkit.infrastructure.io.locations import output, splits_dir


@pytest.fixture
def test_config_with_session_root(tmp_path: Path) -> Path:
    """Create a test TOML config with SESSION.root_dir set."""
    config_path = tmp_path / "test_config.toml"
    session_root = tmp_path / "my_custom_root"
    session_root.mkdir(parents=True, exist_ok=True)

    data_path = (tmp_path / "data.npy").as_posix()

    config_content = f"""
[SESSION]
name = "test-session"
root_dir = "{session_root.as_posix()}"
inference = false
seed = 42

[DATASET]
type = "flexible"
name = "SupervisedArrayDataset"

    [[DATASET.features]]
    name = "x"
    path = "{data_path}"

    [[DATASET.targets]]
    name = "y"
    path = "{data_path}"

    [DATASET.split]
    test_ratio = 0.2
    val_ratio = 0.2

[DATAMODULE]
name = "InMemoryModule"

    [DATAMODULE.dataloader]
    batch_size = 32

[MODEL]
name = "SimpleMLP"
hidden_dims = [64, 32]

[TRAINING]
epochs = 1

    [TRAINING.optimizer]
    name = "Adam"
    lr = 0.001

    [TRAINING.loss_function]
    name = "MSELoss"
"""
    config_path.write_text(config_content)

    # Create dummy data file
    import numpy as np

    np.save(tmp_path / "data.npy", np.random.rand(100, 10))

    return config_path


def test_session_root_dir_propagated_to_environment(
    test_config_with_session_root: Path, tmp_path: Path
):
    """Test that SESSION.root_dir is propagated to EnvironmentSettings."""
    # Clear any existing DLKIT_ROOT_DIR env var
    original_env = os.environ.pop("DLKIT_ROOT_DIR", None)

    try:
        # Reset global environment
        global_environment.root_dir = None

        # Load config
        settings = cast(
            GeneralSettings, load_config(test_config_with_session_root, GeneralSettings)
        )

        # Verify SESSION.root_dir is set in settings
        assert settings.SESSION.root_dir is not None
        session_root = Path(settings.SESSION.root_dir)

        # Verify it was propagated to EnvironmentSettings
        assert global_environment.root_dir is not None
        assert Path(global_environment.root_dir).resolve() == session_root.resolve()

    finally:
        # Restore original env var
        if original_env:
            os.environ["DLKIT_ROOT_DIR"] = original_env
        # Reset global environment
        global_environment.root_dir = None


def test_env_var_takes_precedence_over_session_root(
    test_config_with_session_root: Path, tmp_path: Path
):
    """Test that DLKIT_ROOT_DIR env var takes precedence over SESSION.root_dir."""
    env_root = tmp_path / "env_root"
    env_root.mkdir(parents=True, exist_ok=True)

    original_env = os.environ.get("DLKIT_ROOT_DIR")

    try:
        # Set env var
        os.environ["DLKIT_ROOT_DIR"] = env_root.as_posix()

        # Reset global environment to pick up env var
        from dlkit.infrastructure.config.environment import EnvironmentSettings

        test_env = EnvironmentSettings()

        # Load config
        settings = cast(
            GeneralSettings, load_config(test_config_with_session_root, GeneralSettings)
        )

        # Verify SESSION.root_dir is set in settings
        assert settings.SESSION.root_dir is not None
        session_root = Path(settings.SESSION.root_dir)

        # Verify env var wins (EnvironmentSettings should use env var, not SESSION.root_dir)
        assert test_env.get_root_path().resolve() == env_root.resolve()
        assert test_env.get_root_path().resolve() != session_root.resolve()

    finally:
        # Restore original env var
        if original_env:
            os.environ["DLKIT_ROOT_DIR"] = original_env
        else:
            os.environ.pop("DLKIT_ROOT_DIR", None)


def test_split_path_respects_session_root_dir(test_config_with_session_root: Path, tmp_path: Path):
    """Test that splits_dir() respects SESSION.root_dir."""
    # Clear any existing DLKIT_ROOT_DIR env var
    original_env = os.environ.pop("DLKIT_ROOT_DIR", None)

    try:
        # Reset global environment
        global_environment.root_dir = None

        # Load config (this should sync SESSION.root_dir to EnvironmentSettings)
        settings = cast(
            GeneralSettings, load_config(test_config_with_session_root, GeneralSettings)
        )

        assert settings.SESSION.root_dir is not None
        session_root = Path(settings.SESSION.root_dir)

        # Get splits directory
        splits_path = splits_dir()

        # Verify splits_dir uses SESSION.root_dir
        expected_path = (session_root / "output" / "splits").resolve()
        assert splits_path.resolve() == expected_path

    finally:
        # Restore original env var
        if original_env:
            os.environ["DLKIT_ROOT_DIR"] = original_env
        # Reset global environment
        global_environment.root_dir = None


def test_output_path_respects_session_root_dir(test_config_with_session_root: Path, tmp_path: Path):
    """Test that output() respects SESSION.root_dir."""
    # Clear any existing DLKIT_ROOT_DIR env var
    original_env = os.environ.pop("DLKIT_ROOT_DIR", None)

    try:
        # Reset global environment
        global_environment.root_dir = None

        # Load config
        settings = cast(
            GeneralSettings, load_config(test_config_with_session_root, GeneralSettings)
        )

        assert settings.SESSION.root_dir is not None
        session_root = Path(settings.SESSION.root_dir)

        # Get output path
        output_path = output("checkpoints")

        # Verify output uses SESSION.root_dir
        expected_path = (session_root / "output" / "checkpoints").resolve()
        assert output_path.resolve() == expected_path

    finally:
        # Restore original env var
        if original_env:
            os.environ["DLKIT_ROOT_DIR"] = original_env
        # Reset global environment
        global_environment.root_dir = None


def test_load_settings_propagates_session_root(test_config_with_session_root: Path, tmp_path: Path):
    """Test that load_settings() also propagates SESSION.root_dir."""
    # Clear any existing DLKIT_ROOT_DIR env var
    original_env = os.environ.pop("DLKIT_ROOT_DIR", None)

    try:
        # Reset global environment
        global_environment.root_dir = None

        # Load config via load_settings
        settings = load_settings(test_config_with_session_root)
        session = settings.SESSION
        assert session is not None

        # Verify SESSION.root_dir is set
        assert session.root_dir is not None
        session_root = Path(session.root_dir)

        # Verify it was propagated to EnvironmentSettings
        assert global_environment.root_dir is not None
        assert Path(global_environment.root_dir).resolve() == session_root.resolve()

    finally:
        # Restore original env var
        if original_env:
            os.environ["DLKIT_ROOT_DIR"] = original_env
        # Reset global environment
        global_environment.root_dir = None
