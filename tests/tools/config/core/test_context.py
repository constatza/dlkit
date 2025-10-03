"""Tests for core.context module.

This module tests the BuildContext class which provides dependency injection
and environment information for object construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from hypothesis import given, strategies as st

from dlkit.tools.config.core.context import BuildContext


# Constants for testing
TEST_MODE = "testing"
TEST_DEVICE = "cpu"
TEST_SEED = 42
TEST_WORKING_DIR = Path("/tmp/test")
TEST_CHECKPOINT = Path("/tmp/model.ckpt")


class TestBuildContextClass:
    """Test suite for BuildContext functionality."""

    def test_build_context_initialization_minimal(self) -> None:
        """Test BuildContext can be initialized with minimal required fields."""
        context = BuildContext(mode=TEST_MODE)

        assert context.mode == TEST_MODE
        assert context.device == "auto"  # Default value
        assert context.random_seed is None  # Default value
        assert context.working_directory == Path.cwd()  # Default factory
        assert context.checkpoint_path is None  # Default value
        assert context.overrides == {}  # Default factory

    def test_build_context_initialization_complete(
        self, build_context_data: dict[str, Any]
    ) -> None:
        """Test BuildContext initialization with all fields specified.

        Args:
            build_context_data: Complete build context dataflow fixture
        """
        context = BuildContext(
            mode=build_context_data["mode"],
            device=build_context_data["device"],
            random_seed=build_context_data["random_seed"],
            working_directory=Path("/tmp/custom"),
            checkpoint_path=TEST_CHECKPOINT,
            overrides=build_context_data["overrides"],
        )

        assert context.mode == build_context_data["mode"]
        assert context.device == build_context_data["device"]
        assert context.random_seed == build_context_data["random_seed"]
        assert context.working_directory == Path("/tmp/custom")
        assert context.checkpoint_path == TEST_CHECKPOINT
        assert context.overrides == build_context_data["overrides"]

    def test_build_context_with_overrides_merges_correctly(
        self, build_context_data: dict[str, Any]
    ) -> None:
        """Test with_overrides merges new overrides with existing ones.

        Args:
            build_context_data: Build context dataflow fixture with existing overrides
        """
        original_context = BuildContext(mode=TEST_MODE, overrides=build_context_data["overrides"])

        new_overrides = {"additional_key": "additional_value", "another_key": 999}
        updated_context = original_context.with_overrides(**new_overrides)

        # Original context should be unchanged
        assert original_context.overrides == build_context_data["overrides"]

        # New context should have merged overrides
        expected_overrides = {**build_context_data["overrides"], **new_overrides}
        assert updated_context.overrides == expected_overrides
        assert updated_context.mode == original_context.mode  # Other fields preserved

    def test_build_context_with_overrides_overwrites_existing_keys(self) -> None:
        """Test with_overrides overwrites existing keys with new values."""
        original_context = BuildContext(
            mode=TEST_MODE, overrides={"key1": "original_value", "key2": 42}
        )

        new_overrides = {"key1": "new_value", "key3": "additional_value"}
        updated_context = original_context.with_overrides(**new_overrides)

        expected_overrides = {
            "key1": "new_value",  # Overwritten
            "key2": 42,  # Preserved
            "key3": "additional_value",  # Added
        }
        assert updated_context.overrides == expected_overrides

    def test_get_override_returns_existing_value(self) -> None:
        """Test get_override returns existing override value."""
        context = BuildContext(
            mode=TEST_MODE, overrides={"test_key": "test_value", "number_key": 123}
        )

        assert context.get_override("test_key") == "test_value"
        assert context.get_override("number_key") == 123

    def test_get_override_returns_default_for_missing_key(self) -> None:
        """Test get_override returns default for non-existent key."""
        context = BuildContext(mode=TEST_MODE, overrides={})

        assert context.get_override("missing_key") is None
        assert context.get_override("missing_key", "default_value") == "default_value"
        assert context.get_override("missing_key", 42) == 42

    def test_build_context_immutability_via_model_copy(self) -> None:
        """Test BuildContext follows immutable pattern through model_copy."""
        original = BuildContext(mode=TEST_MODE, device=TEST_DEVICE, overrides={"key": "value"})

        # Create modified copy
        modified = original.model_copy(
            update={"mode": "new_mode", "overrides": {"key": "new_value"}}
        )

        # Original should be unchanged
        assert original.mode == TEST_MODE
        assert original.overrides == {"key": "value"}

        # Modified should have updates
        assert modified.mode == "new_mode"
        assert modified.overrides == {"key": "new_value"}
        assert modified.device == TEST_DEVICE  # Unchanged field preserved

    def test_build_context_arbitrary_types_allowed(self) -> None:
        """Test BuildContext allows arbitrary types in overrides."""
        custom_object = {"nested": {"deep": "value"}}
        path_object = Path("/custom/path")

        context = BuildContext(
            mode=TEST_MODE,
            working_directory=path_object,
            overrides={
                "path_override": Path("/another/path"),
                "dict_override": custom_object,
                "callable_override": lambda x: x * 2,
            },
        )

        assert isinstance(context.working_directory, Path)
        assert isinstance(context.get_override("path_override"), Path)
        assert context.get_override("dict_override") == custom_object
        assert callable(context.get_override("callable_override"))

    @given(
        st.text(min_size=1, max_size=50),
        st.text(min_size=1, max_size=50),
        st.one_of(st.none(), st.integers(min_value=0, max_value=9999)),
    )
    def test_build_context_property_valid_initialization(
        self, mode: str, device: str, seed: int | None
    ) -> None:
        """Property test: BuildContext accepts valid initialization parameters.

        Args:
            mode: Generated mode string
            device: Generated device string
            seed: Generated seed value or None
        """
        context = BuildContext(mode=mode, device=device, random_seed=seed)

        assert context.mode == mode
        assert context.device == device
        assert context.random_seed == seed

    @given(st.dictionaries(st.text(min_size=1, max_size=20), st.integers()))
    def test_build_context_property_override_operations(
        self, override_data: dict[str, int]
    ) -> None:
        """Property test: Override operations work correctly with generated

        Args:
            override_data: Generated override dictionary
        """
        context = BuildContext(mode=TEST_MODE, overrides=override_data)

        # Test all keys are accessible
        for key, expected_value in override_data.items():
            assert context.get_override(key) == expected_value

        # Test with_overrides preserves original dataflow
        additional_overrides = {"new_key": 999}
        updated_context = context.with_overrides(**additional_overrides)

        # Original dataflow should still be accessible
        for key, expected_value in override_data.items():
            assert updated_context.get_override(key) == expected_value

        # New dataflow should also be accessible
        assert updated_context.get_override("new_key") == 999
