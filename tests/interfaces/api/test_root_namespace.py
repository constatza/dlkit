"""Smoke tests for the dlkit root namespace public API contract."""

from __future__ import annotations

from typing import Any, get_args, get_origin, get_type_hints

import dlkit
from dlkit.common import TrainingResult
from dlkit.interfaces.api import LifecycleHooks


class TestRootNamespaceExports:
    """Verify the root dlkit namespace exposes exactly what the public API contract requires."""

    def test_execute_is_callable_at_root(self) -> None:
        assert callable(dlkit.execute)

    def test_load_model_is_callable_at_root(self) -> None:
        assert callable(dlkit.load_model)

    def test_train_is_callable_at_root(self) -> None:
        assert callable(dlkit.train)

    def test_optimize_is_callable_at_root(self) -> None:
        assert callable(dlkit.optimize)

    def test_curated_config_loaders_are_callable_at_root(self) -> None:
        assert callable(dlkit.load_training_config)
        assert callable(dlkit.load_inference_config)
        assert callable(dlkit.load_optimization_config)

    def test_curated_registry_decorators_are_callable_at_root(self) -> None:
        assert callable(dlkit.register_model)
        assert callable(dlkit.register_dataset)

    def test_legacy_flat_exports_are_not_at_root(self) -> None:
        assert not hasattr(dlkit, "load_config")
        assert not hasattr(dlkit, "validate_config")
        assert not hasattr(dlkit, "load_raw_config")
        assert not hasattr(dlkit, "register_metric")


class TestNamespacedShims:
    """Verify namespaced shims expose the broader surfaces."""

    def test_config_shim_exposes_types_and_loaders(self) -> None:
        from dlkit.config import GeneralSettings, load_training_config

        assert GeneralSettings is not None
        assert callable(load_training_config)

    def test_registry_shim_exposes_introspection(self) -> None:
        from dlkit.registry import describe_model, list_registered_datasets, list_registered_models

        assert callable(list_registered_models)
        assert callable(list_registered_datasets)
        assert callable(describe_model)

    def test_inference_shim_exposes_predictor_api(self) -> None:
        from dlkit.inference import CheckpointPredictor, load_model, validate_checkpoint

        assert CheckpointPredictor is not None
        assert callable(load_model)
        assert callable(validate_checkpoint)


class TestLifecycleHookTyping:
    """Verify the public hook contract carries concrete result types."""

    def test_training_result_types_are_exposed_in_hook_annotations(self) -> None:
        hints = get_type_hints(LifecycleHooks)

        for field_name in (
            "on_training_complete",
            "extra_tags",
            "extra_params",
            "extra_artifacts",
        ):
            callback_type = _unwrap_optional(hints[field_name])
            args = get_args(callback_type)
            assert args
            assert args[0] == [TrainingResult]


def _unwrap_optional(annotation: Any) -> Any:
    if get_origin(annotation) is not None and type(None) in get_args(annotation):
        return next(arg for arg in get_args(annotation) if arg is not type(None))
    return annotation
