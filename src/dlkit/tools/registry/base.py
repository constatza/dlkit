from __future__ import annotations

from threading import RLock


class LockedRegistry[T]:
    """Thread-safe registry with aliasing and an optional forced selection.

    - Stores canonical key → object
    - Supports alias → canonical key resolution
    - Supports a single forced canonical key per registry (set via `set_forced`)
    """

    def __init__(self) -> None:
        self._mapping: dict[str, T] = {}
        self._aliases: dict[str, str] = {}
        self._forced_key: str | None = None
        self._lock = RLock()

    def register(self, key: str, obj: T, *, overwrite: bool = False) -> None:
        with self._lock:
            if not overwrite and key in self._mapping:
                raise ValueError(
                    f"Key '{key}' is already registered; pass overwrite=True to replace it."
                )
            self._mapping[key] = obj

    def add_alias(self, alias: str, canonical: str, *, overwrite: bool = False) -> None:
        with self._lock:
            if canonical not in self._mapping:
                raise KeyError(
                    f"Cannot create alias '{alias}' → '{canonical}': canonical not found"
                )
            if not overwrite and alias in self._aliases and self._aliases[alias] != canonical:
                raise ValueError(
                    f"Alias '{alias}' is already mapped to '{self._aliases[alias]}'; overwrite=True to replace."
                )
            self._aliases[alias] = canonical

    def _canonical_key(self, name: str) -> str | None:
        if name in self._mapping:
            return name
        if name in self._aliases:
            return self._aliases[name]
        return None

    def get(self, name: str) -> T:
        canonical = self._canonical_key(name)
        if canonical is None:
            raise KeyError(name)
        return self._mapping[canonical]

    def has(self, name: str) -> bool:
        return self._canonical_key(name) is not None

    def set_forced(self, name: str) -> None:
        canonical = self._canonical_key(name) or name
        with self._lock:
            if canonical not in self._mapping:
                raise KeyError(f"Cannot force '{name}': not registered")
            self._forced_key = canonical

    def clear_forced(self) -> None:
        with self._lock:
            self._forced_key = None

    def get_forced(self) -> T | None:
        key = self._forced_key
        if key is None:
            return None
        return self._mapping.get(key)

    # Test-only helpers (not exported)
    def _reset_for_tests(self) -> None:
        with self._lock:
            self._mapping.clear()
            self._aliases.clear()
            self._forced_key = None

    # Introspection for suggestions (kept internal)
    def _all_keys(self) -> set[str]:
        return set(self._mapping.keys()) | set(self._aliases.keys())
