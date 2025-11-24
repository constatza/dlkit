"""Tests for DataEntry validation logic.

Ensures that XOR validation for path/value works correctly,
especially when loading from config files where path may be None.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dlkit.tools.config.data_entries import Feature, Target


def test_feature_with_path_valid():
    """Test Feature with valid path."""
    # Using a Path object directly (will be validated by Pydantic)
    from pathlib import Path

    # Create a temp file path (doesn't need to exist for this validation test)
    feat = Feature(name="test", path="/tmp/test.npy")
    assert feat.has_path()
    assert not feat.has_value()


def test_feature_with_value_valid():
    """Test Feature with valid value."""
    arr = np.ones((10, 5), dtype=np.float32)
    feat = Feature(name="test", value=arr)
    assert feat.has_value()
    assert not feat.has_path()


def test_feature_missing_both_path_and_value():
    """Test Feature fails validation when both path and value are missing."""
    with pytest.raises(ValueError, match="must have either 'path' or 'value' specified"):
        Feature(name="test")  # Neither path nor value


def test_feature_with_path_none_explicit():
    """Test Feature fails validation when path is explicitly None."""
    with pytest.raises(ValueError, match="must have either 'path' or 'value' specified"):
        Feature(name="test", path=None)


def test_feature_with_both_path_and_value():
    """Test Feature fails validation when both path and value are provided."""
    arr = np.ones((10, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="cannot have both 'path' and 'value'"):
        Feature(name="test", path="/tmp/test.npy", value=arr)


def test_target_with_path_valid():
    """Test Target with valid path."""
    targ = Target(name="test", path="/tmp/test.npy")
    assert targ.has_path()
    assert not targ.has_value()


def test_target_with_value_valid():
    """Test Target with valid value."""
    arr = np.ones((10, 1), dtype=np.float32)
    targ = Target(name="test", value=arr)
    assert targ.has_value()
    assert not targ.has_path()


def test_target_missing_both_path_and_value():
    """Test Target fails validation when both path and value are missing."""
    with pytest.raises(ValueError, match="must have either 'path' or 'value' specified"):
        Target(name="test")  # Neither path nor value


def test_target_with_path_none_explicit():
    """Test Target fails validation when path is explicitly None."""
    with pytest.raises(ValueError, match="must have either 'path' or 'value' specified"):
        Target(name="test", path=None)


def test_target_with_both_path_and_value():
    """Test Target fails validation when both path and value are provided."""
    arr = np.ones((10, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="cannot have both 'path' and 'value'"):
        Target(name="test", path="/tmp/test.npy", value=arr)


def test_validation_error_messages_include_name():
    """Test that validation error messages include the entry name for debugging."""
    with pytest.raises(ValueError, match="Feature 'my_feature'"):
        Feature(name="my_feature", path=None)

    with pytest.raises(ValueError, match="Target 'my_target'"):
        Target(name="my_target", path=None)


def test_validation_error_messages_helpful():
    """Test that validation error messages provide helpful guidance."""
    with pytest.raises(ValueError, match="Config files should specify"):
        Feature(name="test")

    with pytest.raises(ValueError, match="For programmatic/testing use"):
        Target(name="test")


def test_feature_dict_representation_without_path():
    """Test creating Feature from dict without path field."""
    # This simulates what happens when loading from a config file
    # where the path field is missing
    with pytest.raises(ValueError, match="must have either 'path' or 'value'"):
        Feature.model_validate({"name": "test", "transforms": []})


def test_torch_tensor_value():
    """Test Feature/Target accept torch.Tensor as value."""
    tensor = torch.randn(10, 5, dtype=torch.float32)

    feat = Feature(name="test", value=tensor)
    assert feat.has_value()
    assert isinstance(feat.value, torch.Tensor)

    targ = Target(name="test", value=tensor)
    assert targ.has_value()
    assert isinstance(targ.value, torch.Tensor)
