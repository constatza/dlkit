from collections.abc import Callable
from typing import cast, overload


class Registry[T]:
    """A simple registry for mapping string keys to classes or callables."""

    def __init__(self) -> None:
        self._registry: dict[str, T] = {}

    @overload
    def register[U](self, item: U, *, name: str | None = None) -> U: ...

    @overload
    def register[U](self, item: None = None, *, name: str | None = None) -> Callable[[U], U]: ...

    def register[U](
        self, item: U | None = None, *, name: str | None = None
    ) -> Callable[[U], U] | U:
        """Register an item under `name` or its own __name__ by default.

        Usage:
           @registry.register
           class Foo: ...

           @registry.register(name="custom_name")
           def bar(...): ...

           registry.register(SomeClass, name="explicit")
        """

        def _do_register(target: U) -> U:
            reg_name = name or getattr(target, "__name__", None)
            if reg_name is None:
                raise ValueError("Cannot infer name for registry key")
            self._registry[reg_name] = cast(T, target)
            return target

        # Direct call: registry.register(SomeClass, name="...")
        if item is not None:
            return _do_register(item)
        # Decorator factory: @registry.register or @registry.register(name="...")
        return _do_register

    def get(self, name: str) -> T:
        """Retrieve a registered item by name."""
        try:
            output = self._registry[name]
            return output
        except KeyError as e:
            raise ValueError(f"No entry registered under '{name}'") from e

    def items(self):
        return self._registry.items()

    def keys(self):
        return self._registry.keys()

    def values(self):
        return self._registry.values()
