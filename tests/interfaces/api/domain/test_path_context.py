"""Test suite for PathContext and resolution functions.

This module provides comprehensive testing for the new explicit PathContext
system introduced in Phase 0 of the path context refactoring.

Test Coverage:
- PathContext factory methods (from_dict, from_settings, from_cli_args, empty)
- PathContext helper methods (has_root_override, merge)
- Resolution functions (resolve_root_dir, resolve_component_path)
- Edge cases and error handling
- Integration with EnvironmentSettings
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlkit.infrastructure.config.environment import EnvironmentSettings
from dlkit.infrastructure.config.general_settings import GeneralSettings
from dlkit.infrastructure.config.session_settings import SessionSettings
from dlkit.infrastructure.io.path_context import (
    PathContext,
    resolve_component_path,
    resolve_root_dir,
)

# ============================================================================
# Factory Methods Tests
# ============================================================================


class TestPathContextFromDict:
    """Test PathContext.from_dict() factory method."""

    def test_from_dict_with_all_fields(self) -> None:
        """Test creating PathContext from dict with all fields."""
        overrides = {
            "root_dir": "/project",
            "output_dir": "/output",
            "data_dir": "/data",
            "checkpoints_dir": "/checkpoints",
        }

        ctx = PathContext.from_dict(overrides)

        assert ctx.root_dir == Path("/project")
        assert ctx.output_dir == Path("/output")
        assert ctx.data_dir == Path("/data")
        assert ctx.checkpoints_dir == Path("/checkpoints")

    def test_from_dict_with_partial_fields(self) -> None:
        """Test creating PathContext from dict with only some fields."""
        overrides = {"root_dir": "/project", "output_dir": "/output"}

        ctx = PathContext.from_dict(overrides)

        assert ctx.root_dir == Path("/project")
        assert ctx.output_dir == Path("/output")
        assert ctx.data_dir is None
        assert ctx.checkpoints_dir is None

    def test_from_dict_with_empty_dict(self) -> None:
        """Test creating PathContext from empty dict."""
        ctx = PathContext.from_dict({})

        assert ctx.root_dir is None
        assert ctx.output_dir is None
        assert ctx.data_dir is None
        assert ctx.checkpoints_dir is None

    def test_from_dict_ignores_unknown_keys(self) -> None:
        """Test that from_dict ignores unknown keys."""
        overrides = {
            "root_dir": "/project",
            "unknown_field": "value",
            "another_field": 123,
        }

        ctx = PathContext.from_dict(overrides)

        assert ctx.root_dir == Path("/project")
        assert not hasattr(ctx, "unknown_field")

    def test_from_dict_with_path_objects(self) -> None:
        """Test from_dict accepts Path objects as well as strings."""
        overrides = {
            "root_dir": Path("/project"),
            "output_dir": "/output",  # Mixed str and Path
        }

        ctx = PathContext.from_dict(overrides)

        assert ctx.root_dir == Path("/project")
        assert ctx.output_dir == Path("/output")


class TestPathContextFromSettings:
    """Test PathContext.from_settings() factory method."""

    def test_from_settings_with_root_dir(self) -> None:
        """Test creating PathContext from settings with SESSION.root_dir."""
        settings = GeneralSettings(SESSION=SessionSettings(root_dir="/project"))

        ctx = PathContext.from_settings(settings)

        assert ctx.root_dir == Path("/project")
        assert ctx.output_dir is None
        assert ctx.data_dir is None
        assert ctx.checkpoints_dir is None

    def test_from_settings_without_root_dir(self) -> None:
        """Test creating PathContext from settings without SESSION.root_dir."""
        settings = GeneralSettings(SESSION=SessionSettings())

        ctx = PathContext.from_settings(settings)

        assert ctx.root_dir is None
        assert ctx.output_dir is None

    def test_from_settings_without_session_section(self) -> None:
        """Test creating PathContext from settings without SESSION section."""
        settings = GeneralSettings()

        ctx = PathContext.from_settings(settings)

        assert ctx.root_dir is None


class TestPathContextFromCliArgs:
    """Test PathContext.from_cli_args() factory method."""

    def test_from_cli_args_with_all_args(self) -> None:
        """Test creating PathContext from all CLI args."""
        ctx = PathContext.from_cli_args(
            root_dir="/project",
            output_dir="/output",
            data_dir="/data",
            checkpoints_dir="/checkpoints",
        )

        assert ctx.root_dir == Path("/project")
        assert ctx.output_dir == Path("/output")
        assert ctx.data_dir == Path("/data")
        assert ctx.checkpoints_dir == Path("/checkpoints")

    def test_from_cli_args_with_partial_args(self) -> None:
        """Test creating PathContext from partial CLI args."""
        ctx = PathContext.from_cli_args(root_dir="/project", output_dir="/output")

        assert ctx.root_dir == Path("/project")
        assert ctx.output_dir == Path("/output")
        assert ctx.data_dir is None
        assert ctx.checkpoints_dir is None

    def test_from_cli_args_with_no_args(self) -> None:
        """Test creating PathContext from CLI with no args."""
        ctx = PathContext.from_cli_args()

        assert ctx.root_dir is None
        assert ctx.output_dir is None
        assert ctx.data_dir is None
        assert ctx.checkpoints_dir is None

    def test_from_cli_args_with_path_objects(self) -> None:
        """Test from_cli_args accepts Path objects."""
        ctx = PathContext.from_cli_args(root_dir=Path("/project"), output_dir=Path("/output"))

        assert ctx.root_dir == Path("/project")
        assert ctx.output_dir == Path("/output")


class TestPathContextEmpty:
    """Test PathContext.empty() factory method."""

    def test_empty_creates_context_with_no_overrides(self) -> None:
        """Test that empty() creates context with all None fields."""
        ctx = PathContext.empty()

        assert ctx.root_dir is None
        assert ctx.output_dir is None
        assert ctx.data_dir is None
        assert ctx.checkpoints_dir is None

    def test_empty_is_equivalent_to_default_constructor(self) -> None:
        """Test that empty() is equivalent to PathContext()."""
        ctx1 = PathContext.empty()
        ctx2 = PathContext()

        assert ctx1 == ctx2


# ============================================================================
# Helper Methods Tests
# ============================================================================


class TestPathContextHasRootOverride:
    """Test PathContext.has_root_override() method."""

    def test_has_root_override_returns_true_when_set(self) -> None:
        """Test has_root_override returns True when root_dir is set."""
        ctx = PathContext(root_dir=Path("/project"))

        assert ctx.has_root_override() is True

    def test_has_root_override_returns_false_when_not_set(self) -> None:
        """Test has_root_override returns False when root_dir is None."""
        ctx = PathContext()

        assert ctx.has_root_override() is False

    def test_has_root_override_ignores_other_fields(self) -> None:
        """Test has_root_override only checks root_dir field."""
        ctx = PathContext(output_dir=Path("/output"))

        assert ctx.has_root_override() is False


class TestPathContextMerge:
    """Test PathContext.merge() method."""

    def test_merge_other_takes_precedence(self) -> None:
        """Test that merge gives precedence to other context."""
        base = PathContext(root_dir=Path("/base"), output_dir=Path("/base/output"))
        override = PathContext(output_dir=Path("/custom/output"))

        merged = base.merge(override)

        assert merged.root_dir == Path("/base")  # From base
        assert merged.output_dir == Path("/custom/output")  # From override

    def test_merge_preserves_none_values_correctly(self) -> None:
        """Test merge preserves None values from self when other is None."""
        base = PathContext(root_dir=Path("/base"), output_dir=Path("/output"))
        override = PathContext(data_dir=Path("/data"))

        merged = base.merge(override)

        assert merged.root_dir == Path("/base")
        assert merged.output_dir == Path("/output")
        assert merged.data_dir == Path("/data")
        assert merged.checkpoints_dir is None

    def test_merge_with_empty_context(self) -> None:
        """Test merging with empty context returns copy of self."""
        base = PathContext(root_dir=Path("/base"), output_dir=Path("/output"))
        empty = PathContext.empty()

        merged = base.merge(empty)

        assert merged.root_dir == Path("/base")
        assert merged.output_dir == Path("/output")

    def test_merge_does_not_mutate_original(self) -> None:
        """Test that merge creates new instance, doesn't mutate originals."""
        base = PathContext(root_dir=Path("/base"))
        override = PathContext(root_dir=Path("/override"))

        merged = base.merge(override)

        # Original contexts unchanged
        assert base.root_dir == Path("/base")
        assert override.root_dir == Path("/override")
        # New context has merged values
        assert merged.root_dir == Path("/override")


# ============================================================================
# Immutability Tests
# ============================================================================


class TestPathContextImmutability:
    """Test that PathContext is immutable (frozen dataclass)."""

    def test_cannot_modify_fields(self) -> None:
        """Test that PathContext fields cannot be modified after creation."""
        ctx = PathContext(root_dir=Path("/project"))

        with pytest.raises(AttributeError):
            ctx.root_dir = Path("/new")  # type: ignore

    def test_is_hashable(self) -> None:
        """Test that PathContext is hashable (can be dict key, set member)."""
        ctx1 = PathContext(root_dir=Path("/project"))
        ctx2 = PathContext(root_dir=Path("/project"))

        # Can be used as dict key
        mapping = {ctx1: "value1"}
        assert mapping[ctx2] == "value1"  # Same hash

        # Can be used in set
        contexts = {ctx1, ctx2}
        assert len(contexts) == 1  # Same hash, deduplicated


# ============================================================================
# Resolution Functions Tests
# ============================================================================


class TestResolveRootDir:
    """Test resolve_root_dir() function."""

    def test_resolve_root_dir_with_context_takes_precedence(self, tmp_path: Path) -> None:
        """Test that path_context.root_dir takes highest precedence."""
        ctx = PathContext(root_dir=tmp_path / "context")
        env = EnvironmentSettings(root_dir=(tmp_path / "env").as_posix())

        resolved = resolve_root_dir(path_context=ctx, env=env)

        # Context takes precedence over env
        assert resolved == (tmp_path / "context").resolve()

    def test_resolve_root_dir_with_env_fallback(self, tmp_path: Path) -> None:
        """Test that env is used when context has no root_dir."""
        ctx = PathContext.empty()
        env = EnvironmentSettings(root_dir=(tmp_path / "env").as_posix())

        resolved = resolve_root_dir(path_context=ctx, env=env)

        assert resolved == (tmp_path / "env").resolve()

    def test_resolve_root_dir_with_cwd_fallback(self) -> None:
        """Test that cwd is used when context and env are None."""
        resolved = resolve_root_dir(path_context=None, env=None)

        assert resolved == Path.cwd().resolve()

    def test_resolve_root_dir_returns_absolute_path(self, tmp_path: Path) -> None:
        """Test that resolve_root_dir always returns absolute path."""
        ctx = PathContext(root_dir=Path("relative/path"))

        resolved = resolve_root_dir(path_context=ctx)

        assert resolved.is_absolute()

    def test_resolve_root_dir_with_none_context(self, tmp_path: Path) -> None:
        """Test resolve_root_dir with None context uses env."""
        env = EnvironmentSettings(root_dir=(tmp_path / "env").as_posix())

        resolved = resolve_root_dir(path_context=None, env=env)

        assert resolved == (tmp_path / "env").resolve()


class TestResolveComponentPath:
    """Test resolve_component_path() function."""

    def test_resolve_component_path_with_output_override(self, tmp_path: Path) -> None:
        """Test resolving 'output' with explicit output_dir override."""
        ctx = PathContext(output_dir=tmp_path / "custom_output")

        resolved = resolve_component_path("output", path_context=ctx)

        assert resolved == (tmp_path / "custom_output").resolve()

    def test_resolve_component_path_with_output_prefix(self, tmp_path: Path) -> None:
        """Test resolving 'output/mlruns' with output_dir override."""
        ctx = PathContext(output_dir=tmp_path / "custom_output")

        resolved = resolve_component_path("output/mlruns", path_context=ctx)

        assert resolved == (tmp_path / "custom_output" / "mlruns").resolve()

    def test_resolve_component_path_with_data_override(self, tmp_path: Path) -> None:
        """Test resolving 'data' with explicit data_dir override."""
        ctx = PathContext(data_dir=tmp_path / "custom_data")

        resolved = resolve_component_path("data", path_context=ctx)

        assert resolved == (tmp_path / "custom_data").resolve()

    def test_resolve_component_path_with_checkpoints_override(self, tmp_path: Path) -> None:
        """Test resolving 'checkpoints' with explicit checkpoints_dir override."""
        ctx = PathContext(checkpoints_dir=tmp_path / "custom_checkpoints")

        resolved = resolve_component_path("checkpoints", path_context=ctx)

        assert resolved == (tmp_path / "custom_checkpoints").resolve()

    def test_resolve_component_path_with_checkpoint_substring(self, tmp_path: Path) -> None:
        """Test resolving paths containing 'checkpoint' uses checkpoints_dir."""
        ctx = PathContext(checkpoints_dir=tmp_path / "models")

        resolved = resolve_component_path("checkpoint_best.ckpt", path_context=ctx)

        assert resolved == (tmp_path / "models").resolve()

    def test_resolve_component_path_standard_resolution(self, tmp_path: Path) -> None:
        """Test standard resolution without overrides uses root/component."""
        ctx = PathContext(root_dir=tmp_path)

        resolved = resolve_component_path("output/predictions", path_context=ctx)

        assert resolved == (tmp_path / "output" / "predictions").resolve()

    def test_resolve_component_path_with_no_context(self) -> None:
        """Test resolve_component_path with no context uses cwd."""
        resolved = resolve_component_path("output")

        expected = (Path.cwd() / "output").resolve()
        assert resolved == expected

    def test_resolve_component_path_with_env_fallback(self, tmp_path: Path) -> None:
        """Test resolve_component_path uses env for root when no context."""
        env = EnvironmentSettings(root_dir=tmp_path.as_posix())

        resolved = resolve_component_path("data", env=env)

        assert resolved == (tmp_path / "data").resolve()

    def test_resolve_component_path_returns_absolute(self, tmp_path: Path) -> None:
        """Test that resolve_component_path always returns absolute path."""
        ctx = PathContext(root_dir=tmp_path)

        resolved = resolve_component_path("relative/path", path_context=ctx)

        assert resolved.is_absolute()


# ============================================================================
# Integration Tests
# ============================================================================


class TestPathContextIntegration:
    """Integration tests for PathContext with real scenarios."""

    def test_cli_workflow_scenario(self, tmp_path: Path) -> None:
        """Test typical CLI workflow: create context from CLI args, resolve paths."""
        # Simulate CLI args: --root-dir /project --output-dir /custom/output
        ctx = PathContext.from_cli_args(
            root_dir=tmp_path / "project", output_dir=tmp_path / "custom" / "output"
        )

        # Resolve various paths
        root = resolve_root_dir(path_context=ctx)
        output = resolve_component_path("output", path_context=ctx)
        mlruns = resolve_component_path("output/mlruns", path_context=ctx)
        data = resolve_component_path("data", path_context=ctx)

        assert root == (tmp_path / "project").resolve()
        assert output == (tmp_path / "custom" / "output").resolve()
        assert mlruns == (tmp_path / "custom" / "output" / "mlruns").resolve()
        assert data == (tmp_path / "project" / "data").resolve()

    def test_api_workflow_scenario(self, tmp_path: Path) -> None:
        """Test typical API workflow: create context from dict, merge with settings."""
        # Settings from config
        settings = GeneralSettings(SESSION=SessionSettings(root_dir=(tmp_path / "base").as_posix()))
        base_ctx = PathContext.from_settings(settings)

        # API overrides
        override_ctx = PathContext.from_dict({"output_dir": tmp_path / "api_output"})

        # Merge (API overrides take precedence)
        ctx = base_ctx.merge(override_ctx)

        # Resolve paths
        root = resolve_root_dir(path_context=ctx)
        output = resolve_component_path("output", path_context=ctx)

        assert root == (tmp_path / "base").resolve()
        assert output == (tmp_path / "api_output").resolve()

    def test_precedence_hierarchy(self, tmp_path: Path) -> None:
        """Test full precedence hierarchy: context > env > cwd."""
        # All three sources
        ctx = PathContext(root_dir=tmp_path / "context")
        env = EnvironmentSettings(root_dir=(tmp_path / "env").as_posix())

        # Context wins
        resolved = resolve_root_dir(path_context=ctx, env=env)
        assert resolved == (tmp_path / "context").resolve()

        # No context, env wins
        empty_ctx = PathContext.empty()
        resolved = resolve_root_dir(path_context=empty_ctx, env=env)
        assert resolved == (tmp_path / "env").resolve()

        # No context, no env, cwd wins
        resolved = resolve_root_dir(path_context=None, env=None)
        assert resolved == Path.cwd().resolve()


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestPathContextEdgeCases:
    """Test edge cases and potential errors."""

    def test_context_with_nonexistent_paths(self) -> None:
        """Test that PathContext accepts non-existent paths (validation elsewhere)."""
        ctx = PathContext(
            root_dir=Path("/nonexistent/path"),
            output_dir=Path("/also/nonexistent"),
        )

        # Should not raise, validation is not PathContext's responsibility
        assert ctx.root_dir == Path("/nonexistent/path")
        assert ctx.output_dir == Path("/also/nonexistent")

    def test_resolve_with_relative_paths(self, tmp_path: Path) -> None:
        """Test that resolution converts relative to absolute paths."""
        ctx = PathContext(root_dir=Path("relative/path"))

        resolved = resolve_root_dir(path_context=ctx)

        # Should be absolute (resolved from cwd)
        assert resolved.is_absolute()

    def test_empty_component_path(self, tmp_path: Path) -> None:
        """Test resolving empty component path returns root."""
        ctx = PathContext(root_dir=tmp_path)

        resolved = resolve_component_path("", path_context=ctx)

        assert resolved == tmp_path.resolve()

    def test_multiple_merge_operations(self, tmp_path: Path) -> None:
        """Test chaining multiple merge operations."""
        ctx1 = PathContext(root_dir=tmp_path / "root1")
        ctx2 = PathContext(output_dir=tmp_path / "output2")
        ctx3 = PathContext(data_dir=tmp_path / "data3", output_dir=tmp_path / "output3")

        # Chain merges: ctx1.merge(ctx2).merge(ctx3)
        merged = ctx1.merge(ctx2).merge(ctx3)

        assert merged.root_dir == tmp_path / "root1"  # From ctx1
        assert merged.output_dir == tmp_path / "output3"  # From ctx3 (last)
        assert merged.data_dir == tmp_path / "data3"  # From ctx3


# ============================================================================
# Equality and Comparison Tests
# ============================================================================


class TestPathContextEquality:
    """Test PathContext equality and comparison."""

    def test_contexts_with_same_values_are_equal(self) -> None:
        """Test that contexts with same field values are equal."""
        ctx1 = PathContext(root_dir=Path("/project"))
        ctx2 = PathContext(root_dir=Path("/project"))

        assert ctx1 == ctx2

    def test_contexts_with_different_values_are_not_equal(self) -> None:
        """Test that contexts with different field values are not equal."""
        ctx1 = PathContext(root_dir=Path("/project1"))
        ctx2 = PathContext(root_dir=Path("/project2"))

        assert ctx1 != ctx2

    def test_empty_contexts_are_equal(self) -> None:
        """Test that empty contexts are equal."""
        ctx1 = PathContext.empty()
        ctx2 = PathContext()

        assert ctx1 == ctx2
