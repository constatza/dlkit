"""Test for config loading when path field is missing or None.

This simulates the error case from the user's config file where
features/targets are defined without the 'path' field.

With the new architecture:
- PathFeature/PathTarget: Allow placeholder mode (path=None) for value injection
- ValueError is raised if path is None AND value is not provided at runtime
- Use PathFeature.model_validate() for config parsing (not Feature factory)
"""

import pytest

from dlkit.tools.config.data_entries import (
    Feature,
    Target,
    PathFeature,
    PathTarget,
)


def test_config_dict_missing_path_field_creates_placeholder():
    """Test that missing 'path' field in config dict creates a placeholder."""
    # This simulates loading from a TOML config where path is not specified
    config_dict = {
        "name": "rhs",
        "dtype": None,
        "transforms": [],
        # path field missing entirely - creates placeholder
    }

    # With new architecture, this creates a placeholder PathFeature
    feat = PathFeature.model_validate(config_dict)
    assert feat.is_placeholder()
    assert feat.name == "rhs"
    assert not feat.has_path()


def test_config_dict_path_none_explicit_creates_placeholder():
    """Test that explicit path=None in config dict creates a placeholder."""
    # This simulates loading from a TOML config with path = None
    config_dict = {
        "name": "matrix",
        "dtype": None,
        "transforms": [],
        "path": None,  # Explicitly None - placeholder mode
    }

    # With new architecture, this creates a placeholder PathFeature
    feat = PathFeature.model_validate(config_dict)
    assert feat.is_placeholder()
    assert feat.name == "matrix"


def test_target_config_dict_missing_path_creates_placeholder():
    """Test that Target with missing path creates a placeholder."""
    config_dict = {
        "name": "solutions",
        "dtype": None,
        "transforms": [],
        # path field missing - creates placeholder
    }

    # With new architecture, this creates a placeholder PathTarget
    targ = PathTarget.model_validate(config_dict)
    assert targ.is_placeholder()
    assert targ.name == "solutions"


def test_factory_function_creates_placeholder():
    """Test that Feature()/Target() factory functions create placeholders when no path/value."""
    feat = Feature(name="test_feature")
    targ = Target(name="test_target")

    # Both should be placeholders (PathFeature/PathTarget without path)
    assert feat.is_placeholder()
    assert targ.is_placeholder()
    assert isinstance(feat, PathFeature)
    assert isinstance(targ, PathTarget)


def test_factory_function_with_path_is_not_placeholder():
    """Test that Feature()/Target() with path are not placeholders.

    Note: Uses model_construct to bypass validation since we're testing
    the placeholder logic, not path existence validation.
    """
    # Use model_construct to bypass path validation
    feat = PathFeature.model_construct(name="test_feature", path="test.npy")
    targ = PathTarget.model_construct(name="test_target", path="test.npy")

    assert not feat.is_placeholder()
    assert not targ.is_placeholder()
    assert feat.has_path()
    assert targ.has_path()


def test_factory_function_both_path_and_value_raises():
    """Test that factory functions raise error when both path and value provided."""
    import numpy as np

    arr = np.ones((10, 5))

    with pytest.raises(ValueError) as exc_info:
        Feature(name="test", path="test.npy", value=arr)

    assert "cannot have both 'path' and 'value'" in str(exc_info.value)
