from dlkit.tools.config.core import BuildContext, FactoryProvider
from dlkit.tools.config.model_components import LossComponentSettings
from dlkit.tools.registry import register_loss
from dlkit.tools.registry.public import _reset_for_tests


def setup_function() -> None:  # pytest hook per-test
    _reset_for_tests()


def test_factory_uses_registry_for_loss_callable():
    def my_simple_loss(x, y):
        return (x, y)

    register_loss(name="my_simple_loss")(my_simple_loss)

    settings = LossComponentSettings(name="my_simple_loss")
    # Create via default factory; for callables it returns the callable, not invoked
    obj = FactoryProvider.create_component(settings, BuildContext(mode="training"))
    assert obj is my_simple_loss
