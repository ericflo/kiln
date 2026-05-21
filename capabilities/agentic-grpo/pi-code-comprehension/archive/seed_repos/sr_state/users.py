"""Top-level user-facing functions that consume cache.*."""

from __future__ import annotations

from cache import register_cache, get_cache, clear_all_caches


def init_user_caches() -> None:
    register_cache("sessions", 1024)
    register_cache("preferences", 256)


def reset_user_caches() -> None:
    clear_all_caches()


def lookup_session(token: str):
    c = get_cache("sessions")
    if c is None:
        return None
    return c.get(token)
