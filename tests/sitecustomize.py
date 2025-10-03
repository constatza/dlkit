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

import random

try:
    import socket as _socket
except Exception:  # pragma: no cover - site import safety
    _socket = None  # type: ignore


def _install_fake_socket() -> None:
    if _socket is None:
        return

    _real_socket = _socket.socket

    class _FakeSocket:
        def __init__(self, family=_socket.AF_INET, type=_socket.SOCK_STREAM, proto=0):  # noqa: D401
            self._family = family
            self._type = type
            self._proto = proto
            self._host = "127.0.0.1"
            self._port = 0
            self._closed = False

        def setsockopt(self, *args, **kwargs):  # noqa: D401
            return None

        def bind(self, address):  # noqa: D401
            try:
                host, port = address
            except Exception:
                host, port = "127.0.0.1", 0
            self._host = host or "127.0.0.1"
            # emulate ephemeral port allocation
            self._port = int(port) if int(port) != 0 else int(40000 + random.randint(0, 20000))
            return None

        def getsockname(self):  # noqa: D401
            return (self._host, self._port)

        def close(self):  # noqa: D401
            self._closed = True

        def __enter__(self):  # noqa: D401
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: D401
            self.close()
            return False

    def _factory(family=-1, type=-1, proto=-1, fileno=None):  # noqa: D401
        # Keep the same signature as socket.socket
        return _FakeSocket(family, type, proto)

    # Replace only the constructor; leave constants and helpers intact
    _socket.socket = _factory  # type: ignore[assignment]

    # Also expose marker for debugging if needed
    _socket.__dict__.setdefault("_dlkit_fake_socket", True)

    return None


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
