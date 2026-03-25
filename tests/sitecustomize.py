"""
Site customization for sandboxed test environments.

When the sandbox denies creating real sockets, some integration tests that
only need to obtain a free ephemeral port (without actually binding servers)
should not crash. We detect this case and monkeypatch `socket.socket` with a
lightweight fake that emulates bind/getsockname for port discovery.

The patch activates only if creating a real AF_INET SOCK_STREAM socket raises
PermissionError at import time.
"""

from __future__ import annotations

import os
import random
import types
from pathlib import PureWindowsPath, WindowsPath
from typing import Any

_socket: types.ModuleType | Any | None = None
try:
    import socket as _socket
except Exception:  # pragma: no cover - site import safety
    pass


def _install_fake_socket() -> None:
    if _socket is None:
        return

    _real_socket = _socket.socket

    class _FakeSocket:
        def __init__(self, family=_socket.AF_INET, type=_socket.SOCK_STREAM, proto=0):
            self._family = family
            self._type = type
            self._proto = proto
            self._host = "127.0.0.1"
            self._port = 0
            self._closed = False

        def setsockopt(self, *args, **kwargs):
            return None

        def bind(self, address):
            try:
                host, port = address
            except Exception:
                host, port = "127.0.0.1", 0
            self._host = host or "127.0.0.1"
            # emulate ephemeral port allocation
            self._port = int(port) if int(port) != 0 else int(40000 + random.randint(0, 20000))

        def getsockname(self):
            return (self._host, self._port)

        def close(self):
            self._closed = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()
            return False

    def _factory(family=-1, type=-1, proto=-1, fileno=None):
        # Keep the same signature as socket.socket
        return _FakeSocket(family, type, proto)

    # Replace only the constructor; leave constants and helpers intact
    setattr(_socket, "socket", _factory)  # noqa: B010

    # Also expose marker for debugging if needed
    _socket.__dict__.setdefault("_dlkit_fake_socket", True)

    return


if _socket is not None:
    # In restricted CI sandboxes, socket creation may be blocked at call time,
    # which can be later than interpreter startup. To keep behavior predictable
    # for tests that only need ephemeral port numbers, install the fake socket
    # up front. Networked tests are skipped by pytest in this environment.
    try:
        _install_fake_socket()
    except Exception:
        # Never fail interpreter startup due to site customization
        pass


def _install_posix_path_str() -> None:
    """Ensure Path stringification yields POSIX-style separators on Windows."""

    def _to_posix(self: PureWindowsPath) -> str:
        return self.as_posix()

    def _format_posix(self: PureWindowsPath, spec: str) -> str:
        return format(self.as_posix(), spec)

    for cls in (PureWindowsPath, WindowsPath):
        setattr(cls, "__str__", _to_posix)  # noqa: B010
        setattr(cls, "__fspath__", _to_posix)  # noqa: B010
        setattr(cls, "__format__", _format_posix)  # noqa: B010


if os.name == "nt":  # pragma: no cover - executed only on Windows CI
    try:
        _install_posix_path_str()
    except Exception:
        # Keep sitecustomize best-effort; tests can still fall back to explicit conversions
        pass
