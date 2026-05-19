"""Small bounded-size LRU cache with a global registry."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Optional


_REGISTRY: Dict[str, "Cache"] = {}


class Cache:
    """Bounded LRU cache. Capacity must be positive."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._store: "OrderedDict[str, Any]" = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: str, value: Any) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)


def register_cache(name: str, capacity: int) -> Cache:
    """Construct a Cache and register it globally by name.

    Raises ValueError if name is already registered.
    """
    global _REGISTRY
    if name in _REGISTRY:
        raise ValueError(f"cache {name!r} already registered")
    c = Cache(capacity)
    _REGISTRY[name] = c
    return c


def get_cache(name: str) -> Optional[Cache]:
    """Look up a previously registered cache by name. Returns None if missing."""
    return _REGISTRY.get(name)


def clear_all_caches() -> None:
    """Empty every registered cache. Mutates the global registry's caches
    in-place but does NOT remove the registry entries themselves."""
    for c in _REGISTRY.values():
        c._store.clear()
