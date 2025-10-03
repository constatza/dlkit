"""Explicit directory creation for run- and server-related folders.

Keep all directory creation here to avoid side effects in getters/resolution.
"""

from __future__ import annotations

from collections.abc import Iterable

from dlkit.tools.config.environment import DLKitEnvironment, env as global_env
from . import locations


def _env(env: DLKitEnvironment | None) -> DLKitEnvironment:
    return env or global_env


def ensure_internal_dirs(*, env: DLKitEnvironment | None = None) -> None:
    """Ensure internal DLKit directories exist (e.g., .dlkit)."""
    e = _env(env)
    internal = e.get_root_path() / e.internal_dir
    internal.mkdir(parents=True, exist_ok=True)
    # Log file parent exists as part of internal


def ensure_run_dirs(
    *,
    env: DLKitEnvironment | None = None,
    needs: Iterable[str] = ("predictions", "checkpoints", "figures", "lightning"),
) -> None:
    """Ensure standard output directories exist.

    Args:
        needs: names under output/ to create
    """
    base = locations.output(env=env)
    base.mkdir(parents=True, exist_ok=True)
    for name in needs:
        (base / name).mkdir(parents=True, exist_ok=True)


def ensure_mlflow_local_storage(*, env: DLKitEnvironment | None = None) -> None:
    """Ensure local MLflow storage locations exist (backend and artifacts)."""
    (locations.mlruns_dir(env=env)).mkdir(parents=True, exist_ok=True)
    (locations.mlartifacts_dir(env=env)).mkdir(parents=True, exist_ok=True)
