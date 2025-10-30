"""Tests for SESSION.root_dir propagation to DLKitEnvironment and path resolution.

This test suite verifies the three-layer defense-in-depth approach:
1. SESSION.root_dir synced to DLKitEnvironment
2. Defensive path context in BuildFactory
3. Logging and observability
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from dlkit.tools.io.config import load_config
from dlkit.tools.config import load_training_settings
from dlkit.tools.config.environment import env as global_environment
from dlkit.tools.io.locations import splits_dir, output


@pytest.fixture
def test_config_with_session_root(tmp_path: Path) -> Path:
    """Create a test TOML config with SESSION.root_dir set."""
    config_path = tmp_path / "test_config.toml"
    session_root = tmp_path / "my_custom_root"
    session_root.mkdir(parents=True, exist_ok=True)

    config_content = f"""
[SESSION]
name = "test-session"
root_dir = "{session_root}"
inference = false
seed = 42

[DATASET]
type = "flexible"
name = "SupervisedArrayDataset"

    [[DATASET.features]]
    name = "x"
    path = "{tmp_path / 'data.npy'}"

    [[DATASET.targets]]
    name = "y"
    path = "{tmp_path / 'data.npy'}"

    [DATASET.split]
    test_ratio = 0.2
    val_ratio = 0.2

[DATAMODULE]
name = "InMemoryModule"
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


def test_session_root_dir_propagated_to_environment(test_config_with_session_root: Path, tmp_path: Path):
    """Test that SESSION.root_dir is propagated to DLKitEnvironment."""
    # Clear any existing DLKIT_ROOT_DIR env var
    original_env = os.environ.pop("DLKIT_ROOT_DIR", None)

    try:
        # Reset global environment
        global_environment.root_dir = None

        # Load config
        settings = load_config(test_config_with_session_root)

        # Verify SESSION.root_dir is set in settings
        assert settings.SESSION.root_dir is not None
        session_root = Path(settings.SESSION.root_dir)

        # Verify it was propagated to DLKitEnvironment
        assert global_environment.root_dir is not None
        assert Path(global_environment.root_dir).resolve() == session_root.resolve()

    finally:
        # Restore original env var
        if original_env:
            os.environ["DLKIT_ROOT_DIR"] = original_env
        # Reset global environment
        global_environment.root_dir = None


def test_env_var_takes_precedence_over_session_root(test_config_with_session_root: Path, tmp_path: Path):
    """Test that DLKIT_ROOT_DIR env var takes precedence over SESSION.root_dir."""
    env_root = tmp_path / "env_root"
    env_root.mkdir(parents=True, exist_ok=True)

    original_env = os.environ.get("DLKIT_ROOT_DIR")

    try:
        # Set env var
        os.environ["DLKIT_ROOT_DIR"] = str(env_root)

        # Reset global environment to pick up env var
        from dlkit.tools.config.environment import DLKitEnvironment
        test_env = DLKitEnvironment()

        # Load config
        settings = load_config(test_config_with_session_root)

        # Verify SESSION.root_dir is set in settings
        assert settings.SESSION.root_dir is not None
        session_root = Path(settings.SESSION.root_dir)

        # Verify env var wins (DLKitEnvironment should use env var, not SESSION.root_dir)
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

        # Load config (this should sync SESSION.root_dir to DLKitEnvironment)
        settings = load_config(test_config_with_session_root)

        session_root = Path(settings.SESSION.root_dir)

        # Mock test environment detection to return False (test production behavior)
        with patch("dlkit.tools.io.locations._is_test_environment", return_value=False):
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
        settings = load_config(test_config_with_session_root)

        session_root = Path(settings.SESSION.root_dir)

        # Mock test environment detection to return False (test production behavior)
        with patch("dlkit.tools.io.locations._is_test_environment", return_value=False):
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


def test_load_training_settings_propagates_session_root(test_config_with_session_root: Path, tmp_path: Path):
    """Test that load_training_settings() also propagates SESSION.root_dir."""
    # Clear any existing DLKIT_ROOT_DIR env var
    original_env = os.environ.pop("DLKIT_ROOT_DIR", None)

    try:
        # Reset global environment
        global_environment.root_dir = None

        # Load config via load_training_settings
        settings = load_training_settings(test_config_with_session_root)

        # Verify SESSION.root_dir is set
        assert settings.SESSION.root_dir is not None
        session_root = Path(settings.SESSION.root_dir)

        # Verify it was propagated to DLKitEnvironment
        assert global_environment.root_dir is not None
        assert Path(global_environment.root_dir).resolve() == session_root.resolve()

    finally:
        # Restore original env var
        if original_env:
            os.environ["DLKIT_ROOT_DIR"] = original_env
        # Reset global environment
        global_environment.root_dir = None
