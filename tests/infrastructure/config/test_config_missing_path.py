"""Test for config loading when path field is missing or None.

This simulates the error case from the user's config file where
features/targets are defined without the 'path' field.

With the new architecture:
- NpyEntry/ZarrEntry/NpzEntry allow placeholder mode (path=None) for value injection
- ValueError is raised if path is None AND value is not provided at runtime
- Use NpyEntry.model_validate() for config parsing (not Feature factory)
"""

from dlkit.infrastructure.config.data_roles import DataRole
from dlkit.infrastructure.config.entry_types import NpyEntry, ValueEntry


def test_config_dict_missing_path_field_creates_placeholder():
    """Test that missing 'path' field in config dict creates a placeholder."""
    # This simulates loading from a TOML config where path is not specified
    config_dict = {
        "name": "rhs",
        "dtype": None,
        "transforms": [],
        "data_role": "feature",
        # path field missing entirely - creates placeholder
    }

    # With new architecture, this creates a placeholder NpyEntry
    feat = NpyEntry.model_validate(config_dict)
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
        "data_role": "feature",
        "path": None,  # Explicitly None - placeholder mode
    }

    # With new architecture, this creates a placeholder NpyEntry
    feat = NpyEntry.model_validate(config_dict)
    assert feat.is_placeholder()
    assert feat.name == "matrix"


def test_target_config_dict_missing_path_creates_placeholder():
    """Test that target NpyEntry with missing path creates a placeholder."""
    config_dict = {
        "name": "solutions",
        "dtype": None,
        "transforms": [],
        "data_role": "target",
        # path field missing - creates placeholder
    }

    # With new architecture, this creates a placeholder NpyEntry (target)
    targ = NpyEntry.model_validate(config_dict)
    assert targ.is_placeholder()
    assert targ.name == "solutions"


def test_npy_entry_placeholder_without_path():
    """Test that NpyEntry without path is a placeholder."""
    feat = NpyEntry(name="test_feature", data_role=DataRole.FEATURE)
    targ = NpyEntry(name="test_target", data_role=DataRole.TARGET)

    # Both should be placeholders (NpyEntry without path)
    assert feat.is_placeholder()
    assert targ.is_placeholder()
    assert isinstance(feat, NpyEntry)
    assert isinstance(targ, NpyEntry)


def test_npy_entry_with_path_is_not_placeholder(tmp_path):
    """Test that NpyEntry with path is not a placeholder."""
    import numpy as np

    path = tmp_path / "test.npy"
    np.save(path, np.ones((5, 2)))

    # Use model_construct to bypass validation since we're testing placeholder logic
    feat = NpyEntry.model_construct(name="test_feature", path=path, data_role=DataRole.FEATURE)
    targ = NpyEntry.model_construct(name="test_target", path=path, data_role=DataRole.TARGET)

    assert not feat.is_placeholder()
    assert not targ.is_placeholder()
    assert feat.has_path()
    assert targ.has_path()


def test_value_entry_is_not_placeholder():
    """Test that ValueEntry with value provided is not a placeholder."""
    import numpy as np

    arr = np.ones((10, 5))
    feat = ValueEntry(name="test", value=arr, data_role=DataRole.FEATURE)

    assert not feat.is_placeholder()
    assert feat.has_value()
    assert not feat.has_path()
