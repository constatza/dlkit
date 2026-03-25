"""Integration test for split path resolution with SESSION.root_dir.

This test verifies that splits are saved to the correct location when SESSION.root_dir is set,
covering the bug described in the issue where splits were being saved to CWD instead of
the configured root directory.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from dlkit.tools.config import load_settings
from dlkit.tools.config.environment import env as global_environment
from dlkit.tools.io.locations import splits_dir
from dlkit.tools.io.split_provider import get_or_create_split


@pytest.fixture
def training_config_with_custom_root(tmp_path: Path) -> tuple[Path, Path]:
    """Create a training config with custom SESSION.root_dir.

    Returns:
        Tuple of (config_path, session_root_dir)
    """
    custom_root = tmp_path / "my_project_root"
    custom_root.mkdir(parents=True, exist_ok=True)

    config_path = tmp_path / "training_config.toml"

    # Create dummy data
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_file = data_dir / "train_data.npy"
    np.save(data_file, np.random.rand(100, 10))

    config_content = f"""
[SESSION]
name = "integration-test-session"
root_dir = "{custom_root.as_posix()}"
inference = false
seed = 42
precision = "single"

[DATASET]
type = "flexible"
name = "SupervisedArrayDataset"

    [[DATASET.features]]
    name = "x"
    path = "{data_file.as_posix()}"

    [[DATASET.targets]]
    name = "y"
    path = "{data_file.as_posix()}"

    [DATASET.split]
    test_ratio = 0.15
    val_ratio = 0.15

[DATAMODULE]
name = "InMemoryModule"

    [DATAMODULE.dataloader]
    batch_size = 16
    num_workers = 0
    persistent_workers = false

[MODEL]
name = "SimpleMLP"
hidden_dims = [32, 16]

[TRAINING]
epochs = 1

    [TRAINING.trainer]
    accelerator = "cpu"
    max_epochs = 1

    [TRAINING.optimizer]
    name = "Adam"
    lr = 0.001

    [TRAINING.loss_function]
    name = "MSELoss"
"""
    config_path.write_text(config_content)

    return config_path, custom_root


def test_splits_saved_to_session_root_dir(
    training_config_with_custom_root: tuple[Path, Path], tmp_path: Path
):
    """Test that splits are saved to SESSION.root_dir, not CWD.

    This is the core bug fix: splits should respect SESSION.root_dir.
    """
    config_path, custom_root = training_config_with_custom_root

    # Clear any existing DLKIT_ROOT_DIR env var to test pure SESSION.root_dir behavior
    original_env = os.environ.pop("DLKIT_ROOT_DIR", None)

    try:
        # Reset global environment
        global_environment.root_dir = None

        # Load training settings
        settings = load_settings(config_path)
        session = settings.SESSION
        assert session is not None

        # Verify SESSION.root_dir is set
        assert session.root_dir is not None
        assert Path(session.root_dir).resolve() == custom_root.resolve()

        # Get or create a split
        split = get_or_create_split(
            num_samples=100,
            test_ratio=0.15,
            val_ratio=0.15,
            session_name=session.name,
        )

        # Verify split was created
        assert len(split.train) > 0
        assert len(split.validation) > 0
        assert len(split.test) > 0

        # Verify splits directory uses SESSION.root_dir
        expected_splits_dir = (custom_root / "output" / "splits").resolve()
        actual_splits_dir = splits_dir().resolve()

        assert actual_splits_dir == expected_splits_dir, (
            f"Splits directory mismatch!\n"
            f"Expected: {expected_splits_dir}\n"
            f"Actual: {actual_splits_dir}\n"
            f"SESSION.root_dir: {session.root_dir}"
        )

        # Verify split file was actually created in the correct location
        # Note: filename now includes num_samples to prevent stale cache bugs
        split_file = expected_splits_dir / f"{session.name}_100_split.json"
        assert split_file.exists(), (
            f"Split file not found at expected location: {split_file}\n"
            f"Checked in: {expected_splits_dir}"
        )

        # Verify it's NOT in CWD (the bug we're fixing)
        cwd_split_file = Path.cwd() / "output" / "splits" / f"{session.name}_split.json"
        if cwd_split_file.exists() and cwd_split_file.resolve() != split_file.resolve():
            raise AssertionError(
                f"Split file incorrectly saved to CWD!\n"
                f"CWD location: {cwd_split_file}\n"
                f"Expected location: {split_file}"
            )

    finally:
        # Restore original env var
        if original_env:
            os.environ["DLKIT_ROOT_DIR"] = original_env
        # Reset global environment
        global_environment.root_dir = None


def test_splits_respect_session_root_without_session_name(
    training_config_with_custom_root: tuple[Path, Path], tmp_path: Path
):
    """Test that splits respect SESSION.root_dir even when SESSION.name is default.

    This specifically addresses the user's concern that SESSION.name affects split paths.
    The bug is about the DIRECTORY, not the filename.
    """
    config_path, custom_root = training_config_with_custom_root

    # Clear any existing DLKIT_ROOT_DIR env var
    original_env = os.environ.pop("DLKIT_ROOT_DIR", None)

    try:
        # Reset global environment
        global_environment.root_dir = None

        # Load training settings
        load_settings(config_path)

        # Use default session name
        session_name = "dlkit-session"  # Default value

        # Get or create a split
        get_or_create_split(
            num_samples=100,
            test_ratio=0.15,
            val_ratio=0.15,
            session_name=session_name,
        )

        # Verify splits directory uses SESSION.root_dir (NOT CWD)
        expected_splits_dir = (custom_root / "output" / "splits").resolve()
        actual_splits_dir = splits_dir().resolve()

        assert actual_splits_dir == expected_splits_dir, (
            f"Splits directory should use SESSION.root_dir even with default session name!\n"
            f"Expected: {expected_splits_dir}\n"
            f"Actual: {actual_splits_dir}"
        )

        # Verify split file exists in correct location
        # Note: filename now includes num_samples
        split_file = expected_splits_dir / f"{session_name}_100_split.json"
        assert split_file.exists(), f"Split file not found: {split_file}"

    finally:
        # Restore original env var
        if original_env:
            os.environ["DLKIT_ROOT_DIR"] = original_env
        # Reset global environment
        global_environment.root_dir = None


def test_multiple_splits_with_different_sessions(
    training_config_with_custom_root: tuple[Path, Path], tmp_path: Path
):
    """Test that multiple splits with different session names all go to SESSION.root_dir."""
    config_path, custom_root = training_config_with_custom_root

    # Clear any existing DLKIT_ROOT_DIR env var
    original_env = os.environ.pop("DLKIT_ROOT_DIR", None)

    try:
        # Reset global environment
        global_environment.root_dir = None

        # Load training settings
        load_settings(config_path)

        # Create splits with different session names
        session_names = ["session-a", "session-b", "session-c"]
        expected_splits_dir = (custom_root / "output" / "splits").resolve()

        for session_name in session_names:
            get_or_create_split(
                num_samples=100,
                test_ratio=0.15,
                val_ratio=0.15,
                session_name=session_name,
            )

            # Verify split file exists in correct location
            # Note: filename now includes num_samples
            split_file = expected_splits_dir / f"{session_name}_100_split.json"
            assert split_file.exists(), (
                f"Split file for session '{session_name}' not found at: {split_file}"
            )

        # Verify ALL splits are in SESSION.root_dir, not scattered in CWD
        assert expected_splits_dir.exists()
        split_files = list(expected_splits_dir.glob("*_split.json"))
        assert len(split_files) >= len(session_names), (
            f"Not all split files found in {expected_splits_dir}. "
            f"Expected at least {len(session_names)}, found {len(split_files)}"
        )

    finally:
        # Restore original env var
        if original_env:
            os.environ["DLKIT_ROOT_DIR"] = original_env
        # Reset global environment
        global_environment.root_dir = None
