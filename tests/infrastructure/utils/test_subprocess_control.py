from __future__ import annotations

import types

import pytest

import dlkit.infrastructure.utils.subprocess as sp


class _Proc:
    def __init__(self, pid: int, children: list[_Proc] | None = None):
        self.pid = pid
        self._alive = True
        self._children = children or []

    def is_running(self) -> bool:
        return self._alive

    def children(self, recursive: bool = True):
        return list(self._children)

    def terminate(self) -> None:
        self._alive = False

    def kill(self) -> None:
        self._alive = False


def test_stop_process_tree_graceful(monkeypatch: pytest.MonkeyPatch) -> None:
    # Build a small tree: root->child
    child = _Proc(2)
    root = _Proc(1, [child])

    # Patch psutil.Process() to return our root proc irrespective of pid
    ps = types.SimpleNamespace()
    ps.Process = lambda pid: root
    ps.NoSuchProcess = Exception

    def _wait_procs(procs, timeout=0.1):
        # All terminated successfully (gone) and no alive ones
        return (list(procs), [])

    ps.wait_procs = _wait_procs
    monkeypatch.setattr(sp, "psutil", ps)
    monkeypatch.setattr(sp.time, "sleep", lambda s: None)

    sp.stop_process_tree(pid=1, timeout=0.2, initial_interval=0.01)


def test_stop_process_tree_hard_kill(monkeypatch: pytest.MonkeyPatch) -> None:
    # Tree where terminate doesn't kill processes
    c1 = _Proc(3)
    c2 = _Proc(4)
    root = _Proc(2, [c1, c2])

    ps = types.SimpleNamespace()
    ps.NoSuchProcess = Exception
    ps.Process = lambda pid: root

    # first terminate leaves all alive
    def _wait_terminate(procs, timeout=0.1):
        return ([], list(procs))

    # kill then removes all
    def _wait_kill(procs, timeout=0.1):
        for p in procs:
            p._alive = False
        return (list(procs), [])

    # Swap wait_procs behavior depending on which function calls it
    monkeypatch.setattr(sp, "psutil", ps)
    monkeypatch.setattr(sp.time, "sleep", lambda s: None)

    # Patch terminate_procs/kill_procs to use our different waits
    monkeypatch.setattr(sp, "psutil", ps)
    # Monkeypatch inside the helpers by temporarily replacing wait_procs
    ps.wait_procs = _wait_terminate
    # Run with timeout=0 to skip terminate loop and go to kill
    ps.wait_procs = _wait_kill
    sp.stop_process_tree(pid=2, timeout=0.0)
