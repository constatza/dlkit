import builtins

import pytest

from dlkit.infrastructure.registry import (
    describe_model,
    list_registered_datasets,
    list_registered_models,
    register_datamodule,
    register_dataset,
    register_loss,
    register_metric,
    register_model,
)
from dlkit.infrastructure.registry.public import _reset_for_tests
from dlkit.infrastructure.registry.resolve import resolve_component


def setup_function() -> None:  # pytest hook per-test
    _reset_for_tests()


def test_auto_name_registration_and_resolution_model():
    class MyNet:
        pass

    register_model()(MyNet)

    # Should resolve by registered name
    resolved = resolve_component("model", "MyNet")
    assert resolved is MyNet


def test_alias_and_duplicate_protection_dataset():
    class DataA:
        pass

    class DataB:
        pass

    register_dataset(name="A", aliases=["ax"])(DataA)

    # Re-using alias for a different canonical without overwrite must fail
    with pytest.raises(ValueError):
        register_dataset(name="B", aliases=["ax"])(DataB)


def test_use_flag_forced_precedence_over_config_name():
    class A:
        pass

    class B:
        pass

    register_model(name="A", use=True)(A)
    register_model(name="B")(B)

    # Even if config "asks" for B, forced A wins
    resolved = resolve_component("model", "B")
    assert resolved is A


def test_import_fallback_for_third_party_when_not_registered():
    # Import a stdlib function via fallback
    obj = resolve_component("metric", "pow", module_path="builtins")
    assert obj is builtins.pow


def test_register_loss_and_factory_return_callable():
    # No torch dependency here: simple python callable
    def my_loss(x, y):
        return (x, y)

    register_loss(name="my_loss")(my_loss)

    # Resolver should return our callable directly
    resolved = resolve_component("loss", "my_loss")
    assert resolved is my_loss


def test_register_metric_and_datamodule_basic():
    class MyMetric:
        pass

    class MyDM:
        pass

    register_metric()(MyMetric)
    register_datamodule(use=True)(MyDM)

    assert resolve_component("metric", "MyMetric") is MyMetric
    # Forced selection ignores provided name
    assert resolve_component("datamodule", name="Anything") is MyDM


def test_list_registered_models_returns_sorted_canonical_names():
    class ZedModel:
        pass

    class AlphaModel:
        pass

    register_model(name="zed")(ZedModel)
    register_model(name="alpha", aliases=["a"])(AlphaModel)

    assert list_registered_models() == ["alpha", "zed"]


def test_list_registered_datasets_returns_sorted_canonical_names():
    class DatasetB:
        pass

    class DatasetA:
        pass

    register_dataset(name="dataset_b")(DatasetB)
    register_dataset(name="dataset_a", aliases=["a"])(DatasetA)

    assert list_registered_datasets() == ["dataset_a", "dataset_b"]


def test_describe_model_reports_aliases_and_forced_state():
    class MyModel:
        pass

    register_model(name="MyModel", aliases=["mynet"], use=True)(MyModel)

    entry = describe_model("mynet")

    assert entry.kind == "model"
    assert entry.name == "MyModel"
    assert entry.target is MyModel
    assert entry.aliases == ("mynet",)
    assert entry.module_path == MyModel.__module__
    assert entry.qualname == MyModel.__qualname__
    assert entry.forced is True
