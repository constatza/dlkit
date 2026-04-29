"""Smoke tests for the dlkit root namespace public API contract."""

from __future__ import annotations

import dlkit


class TestRootNamespaceExports:
    """Verify the root dlkit namespace exposes exactly what the public API contract requires."""

    def test_execute_is_callable_at_root(self) -> None:
        assert callable(dlkit.execute)

    def test_load_model_is_callable_at_root(self) -> None:
        assert callable(dlkit.load_model)

    def test_train_is_not_exposed_at_root(self) -> None:
        assert not hasattr(dlkit, "train")

    def test_optimize_is_not_exposed_at_root(self) -> None:
        assert not hasattr(dlkit, "optimize")

    def test_load_config_is_callable_at_root(self) -> None:
        assert callable(dlkit.load_config)

    def test_validate_config_is_callable_at_root(self) -> None:
        assert callable(dlkit.validate_config)
