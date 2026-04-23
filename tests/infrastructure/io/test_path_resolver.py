"""Tests for the unified PathResolver service.

This test suite verifies that PathResolver correctly consolidates path
resolution from multiple sources (thread-local context, environment, config)
with explicit precedence.
"""

import os
from pathlib import Path

import pytest

from dlkit.infrastructure.config.environment import EnvironmentSettings
from dlkit.infrastructure.io.path_context_state import PathOverrideContext
from dlkit.infrastructure.io.path_resolver import PathResolver


@pytest.fixture
def clean_env() -> None:
    """Clean environment for isolation between tests."""
    original_env = os.environ.pop("DLKIT_ROOT_DIR", None)
    yield
    if original_env:
        os.environ["DLKIT_ROOT_DIR"] = original_env


def test_path_resolver_resolve_with_no_inputs(tmp_path: Path) -> None:
    """Test that resolver falls back to CWD when no inputs provided."""
    resolver = PathResolver()
    result = resolver.resolve()
    assert result.is_absolute()
    assert result == Path.cwd().resolve()


def test_path_resolver_thread_local_context_precedence(tmp_path: Path) -> None:
    """Test that thread-local context takes highest precedence."""
    context_root = tmp_path / "context_root"
    context_root.mkdir()

    context = PathOverrideContext(root_dir=context_root)
    resolver = PathResolver(thread_local_context=context)

    result = resolver.get_root()
    assert result.resolve() == context_root.resolve()


def test_path_resolver_env_precedence_over_cwd(tmp_path: Path, clean_env: None) -> None:
    """Test that environment root takes precedence over CWD."""
    env_root = tmp_path / "env_root"
    env_root.mkdir()

    env = EnvironmentSettings(root_dir=str(env_root))
    resolver = PathResolver(environment=env)

    result = resolver.get_root()
    assert result.resolve() == env_root.resolve()


def test_path_resolver_config_root_precedence(tmp_path: Path) -> None:
    """Test that config root has lower precedence than env/context."""
    config_root = tmp_path / "config_root"
    config_root.mkdir()

    resolver = PathResolver(config_root=config_root)
    result = resolver.get_root()
    assert result.resolve() == config_root.resolve()


def test_path_resolver_layered_precedence(tmp_path: Path) -> None:
    """Test complete precedence: context > env > config > cwd."""
    context_root = tmp_path / "context_root"
    context_root.mkdir()
    env_root = tmp_path / "env_root"
    env_root.mkdir()
    config_root = tmp_path / "config_root"
    config_root.mkdir()

    # Only context
    context = PathOverrideContext(root_dir=context_root)
    env = EnvironmentSettings(root_dir=str(env_root))
    resolver = PathResolver(thread_local_context=context, environment=env, config_root=config_root)
    assert resolver.get_root().resolve() == context_root.resolve()

    # No context, env takes precedence
    resolver = PathResolver(environment=env, config_root=config_root)
    assert resolver.get_root().resolve() == env_root.resolve()

    # No context, no env, config takes precedence
    resolver = PathResolver(config_root=config_root)
    assert resolver.get_root().resolve() == config_root.resolve()


def test_path_resolver_resolve_relative_path(tmp_path: Path) -> None:
    """Test that relative paths are resolved relative to root."""
    root = tmp_path / "root"
    root.mkdir()

    resolver = PathResolver(thread_local_context=PathOverrideContext(root_dir=root))
    result = resolver.resolve("data/subdir")
    assert result == (root / "data" / "subdir").resolve()


def test_path_resolver_resolve_absolute_path(tmp_path: Path) -> None:
    """Test that absolute paths bypass root resolution."""
    root = tmp_path / "root"
    root.mkdir()
    absolute_path = tmp_path / "absolute"
    absolute_path.mkdir()

    resolver = PathResolver(thread_local_context=PathOverrideContext(root_dir=root))
    result = resolver.resolve(absolute_path)
    assert result == absolute_path.resolve()


def test_path_resolver_resolve_component_path_output_override(tmp_path: Path) -> None:
    """Test that output_dir override is used for 'output' component."""
    custom_output = tmp_path / "custom_output"
    custom_output.mkdir()

    context = PathOverrideContext(output_dir=custom_output)
    resolver = PathResolver(thread_local_context=context)

    result = resolver.resolve_component_path("output")
    assert result == custom_output.resolve()


def test_path_resolver_resolve_component_path_prefixed_output(tmp_path: Path) -> None:
    """Test that prefixed output paths work correctly."""
    custom_output = tmp_path / "custom_output"
    custom_output.mkdir()

    context = PathOverrideContext(output_dir=custom_output)
    resolver = PathResolver(thread_local_context=context)

    result = resolver.resolve_component_path("output/mlruns")
    assert result == (custom_output / "mlruns").resolve()


def test_path_resolver_resolve_component_path_data_override(tmp_path: Path) -> None:
    """Test that data_dir override is used for 'data' component."""
    custom_data = tmp_path / "custom_data"
    custom_data.mkdir()

    context = PathOverrideContext(data_dir=custom_data)
    resolver = PathResolver(thread_local_context=context)

    result = resolver.resolve_component_path("data")
    assert result == custom_data.resolve()


def test_path_resolver_resolve_component_path_checkpoint_override(tmp_path: Path) -> None:
    """Test that checkpoints_dir override is used for 'checkpoint' component."""
    custom_checkpoints = tmp_path / "custom_checkpoints"
    custom_checkpoints.mkdir()

    context = PathOverrideContext(checkpoints_dir=custom_checkpoints)
    resolver = PathResolver(thread_local_context=context)

    result = resolver.resolve_component_path("checkpoints")
    assert result == custom_checkpoints.resolve()


def test_path_resolver_resolve_component_path_fallback_to_root(tmp_path: Path) -> None:
    """Test that component resolution falls back to root-relative when no override."""
    root = tmp_path / "root"
    root.mkdir()

    context = PathOverrideContext(root_dir=root)
    resolver = PathResolver(thread_local_context=context)

    result = resolver.resolve_component_path("output")
    assert result == (root / "output").resolve()


def test_path_resolver_has_context_override(tmp_path: Path) -> None:
    """Test that has_context_override() correctly identifies context override."""
    context_root = tmp_path / "context_root"
    context_root.mkdir()

    context = PathOverrideContext(root_dir=context_root)
    resolver = PathResolver(thread_local_context=context)
    assert resolver.has_context_override() is True

    resolver_no_context = PathResolver()
    assert resolver_no_context.has_context_override() is False


def test_path_resolver_has_env_override(clean_env: None) -> None:
    """Test that has_env_override() correctly identifies env override."""
    # Without env var
    resolver = PathResolver()
    assert resolver.has_env_override() is False

    # With env var
    os.environ["DLKIT_ROOT_DIR"] = "/custom/root"
    resolver = PathResolver()
    assert resolver.has_env_override() is True


def test_path_resolver_from_defaults() -> None:
    """Test that from_defaults() creates resolver with current context."""
    resolver = PathResolver.from_defaults()
    assert resolver is not None
    # Should be usable even if context is None
    root = resolver.get_root()
    assert root.is_absolute()
