"""Test for config loading when path field is missing or None.

This simulates the error case from the user's config file where
features/targets are defined without the 'path' field.
"""

import pytest
from pydantic import ValidationError

from dlkit.tools.config.data_entries import Feature, Target


def test_config_dict_missing_path_field():
    """Test that missing 'path' field in config dict gives clear error."""
    # This simulates loading from a TOML config where path is not specified
    config_dict = {
        "name": "rhs",
        "dtype": None,
        "transforms": [],
        # path field missing entirely
    }

    with pytest.raises(ValidationError) as exc_info:
        Feature.model_validate(config_dict)

    error_msg = str(exc_info.value)
    assert "rhs" in error_msg
    assert "must have either 'path' or 'value'" in error_msg


def test_config_dict_path_none_explicit():
    """Test that explicit path=None in config dict gives clear error."""
    # This simulates loading from a TOML config with path = None
    config_dict = {
        "name": "matrix",
        "dtype": None,
        "transforms": [],
        "path": None,  # Explicitly None
    }

    with pytest.raises(ValidationError) as exc_info:
        Feature.model_validate(config_dict)

    error_msg = str(exc_info.value)
    assert "matrix" in error_msg
    assert "must have either 'path' or 'value'" in error_msg


def test_target_config_dict_missing_path():
    """Test that Target with missing path gives clear error."""
    config_dict = {
        "name": "solutions",
        "dtype": None,
        "transforms": [],
        # path field missing
    }

    with pytest.raises(ValidationError) as exc_info:
        Target.model_validate(config_dict)

    error_msg = str(exc_info.value)
    assert "solutions" in error_msg
    assert "must have either 'path' or 'value'" in error_msg


def test_error_message_provides_guidance():
    """Test that error message provides helpful guidance."""
    with pytest.raises(ValidationError) as exc_info:
        Feature(name="test_feature")

    error_msg = str(exc_info.value)
    # Should mention config files should use 'path'
    assert "Config files should specify 'path" in error_msg
    # Should mention programmatic use can use 'value'
    assert "For programmatic/testing use" in error_msg
