import builtins
import threading

import pytest

from dlkit.common.errors import ForwardContractError
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
from dlkit.infrastructure.registry.base import LockedRegistry
from dlkit.infrastructure.registry.public import _reset_for_tests
from dlkit.infrastructure.registry.resolve import resolve_component


def setup_function() -> None:  # pytest hook per-test
    _reset_for_tests()


def test_auto_name_registration_and_resolution_model():
    class MyNet:
        class InputSpec:
            model_fields: dict = {}

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
        class InputSpec:
            model_fields: dict = {}

    class B:
        class InputSpec:
            model_fields: dict = {}

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
        class InputSpec:
            model_fields: dict = {}

    class AlphaModel:
        class InputSpec:
            model_fields: dict = {}

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


def test_register_model_without_input_spec_raises():
    class NoSpecModel:
        pass

    with pytest.raises(ForwardContractError, match="must declare an InputSpec"):
        register_model()(NoSpecModel)


def test_register_model_with_mismatched_input_spec_field_raises():
    class MismatchedModel:
        class InputSpec:
            model_fields: dict = {"branch": None}

        def forward(self, x):
            return x

    with pytest.raises(ForwardContractError, match="does not match"):
        register_model()(MismatchedModel)


def test_register_model_accepts_plain_nn_module_with_duck_typed_input_spec():
    """register_model requires the structural contract, not DLKitModule inheritance."""

    class ThirdPartyModel:
        class InputSpec:
            model_fields: dict = {"x": None}

        def forward(self, x):
            return x

    register_model()(ThirdPartyModel)

    assert resolve_component("model", "ThirdPartyModel") is ThirdPartyModel


def test_describe_model_reports_aliases_and_forced_state():
    class MyModel:
        class InputSpec:
            model_fields: dict = {}

    register_model(name="MyModel", aliases=["mynet"], use=True)(MyModel)

    entry = describe_model("mynet")

    assert entry.kind == "model"
    assert entry.name == "MyModel"
    assert entry.target is MyModel
    assert entry.aliases == ("mynet",)
    assert entry.module_path == MyModel.__module__
    assert entry.qualname == MyModel.__qualname__
    assert entry.forced is True


def test_locked_registry_public_accessors_expose_state():
    """LockedRegistry's public accessors (used by public.py) reflect its state."""
    registry = LockedRegistry()
    registry.register("Canonical", "target")
    registry.add_alias("alias", "Canonical")
    registry.set_forced("Canonical")

    assert registry.canonical_key("alias") == "Canonical"
    assert registry.canonical_key("missing") is None
    assert registry.all_keys() == {"Canonical", "alias"}
    assert registry.mapping_snapshot() == {"Canonical": "target"}
    assert registry.aliases_snapshot() == {"alias": "Canonical"}
    assert registry.forced_key == "Canonical"


def test_locked_registry_snapshots_are_copies_not_live_views():
    registry = LockedRegistry()
    registry.register("Canonical", "target")

    mapping = registry.mapping_snapshot()
    mapping["Injected"] = "other"

    assert "Injected" not in registry.mapping_snapshot()


def test_locked_registry_accessors_acquire_the_lock():
    """The public accessors must go through self._lock, not read state unguarded."""
    registry = LockedRegistry()
    registry.register("Canonical", "target")
    registry.add_alias("alias", "Canonical")
    registry.set_forced("Canonical")

    class CountingLock:
        def __init__(self, inner: threading.RLock) -> None:
            self._inner = inner
            self.acquisitions = 0

        def __enter__(self):
            self.acquisitions += 1
            return self._inner.__enter__()

        def __exit__(self, *exc_info):
            return self._inner.__exit__(*exc_info)

    counting_lock = CountingLock(registry._lock)
    registry._lock = counting_lock  # ty: ignore[invalid-assignment]

    accessors = (
        lambda: registry.canonical_key("alias"),
        registry.all_keys,
        registry.mapping_snapshot,
        registry.aliases_snapshot,
        lambda: registry.forced_key,
    )
    for accessor in accessors:
        accessor()

    assert counting_lock.acquisitions == len(accessors)
