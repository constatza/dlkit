"""Centralized, environment-aware path policy (pure, no side effects).

All functions use the unified PathResolver which checks the current path override
context first, then falls back to the EnvironmentSettings root. No directories
are created here.
"""

from __future__ import annotations

import os
from pathlib import Path

from dlkit.infrastructure.config.environment import EnvironmentSettings
from dlkit.infrastructure.io.path_resolver import PathResolver


def root() -> Path:
    """Resolved root directory for DLKit operations.

    Priority:
    1) Thread-local path override context (set by CLI/API)
    2) DLKIT_ROOT_DIR environment variable
    3) EnvironmentSettings.root_dir (from SESSION.root_dir)
    4) Current working directory

    Uses PathResolver to consolidate resolution logic.
    """
    resolver = PathResolver.from_defaults()
    return resolver.get_root()


def output(*parts: str, env: EnvironmentSettings | None = None) -> Path:
    """Resolve a path under the standard output directory.

    Honors path override context for output dir when present.

    Args:
        *parts: Path components to join under output directory
        env: Optional EnvironmentSettings (uses global if None)

    Returns:
        Resolved absolute path under output directory
    """
    from loguru import logger

    resolver = PathResolver.from_defaults()
    base = resolver.resolve_component_path("output")
    result = (base.joinpath(*parts)).resolve()

    # Warn if we're falling back to CWD when SESSION.root_dir might be expected
    cwd = Path.cwd()
    if base.resolve() == (cwd / "output").resolve():
        if not os.environ.get("DLKIT_ROOT_DIR"):
            if not resolver.has_context_override():
                logger.debug(
                    f"Output path resolved to CWD/output: {result}. "
                    "If SESSION.root_dir is set, ensure PathOverrideContext is active."
                )

    return result


def predictions_dir(*, env: EnvironmentSettings | None = None) -> Path:
    return output("predictions", env=env)


def checkpoints_dir(*, env: EnvironmentSettings | None = None) -> Path:
    return output("checkpoints", env=env)


def splits_dir(*, env: EnvironmentSettings | None = None) -> Path:
    return output("splits", env=env)


def figures_dir(*, env: EnvironmentSettings | None = None) -> Path:
    return output("figures", env=env)


def lightning_work_dir(*, env: EnvironmentSettings | None = None) -> Path:
    return output("lightning", env=env)


def mlruns_dir(*, env: EnvironmentSettings | None = None) -> Path:
    return output("mlruns", env=env)


def mlruns_backend_uri(*, env: EnvironmentSettings | None = None) -> str:
    # sqlite file under output/mlruns/mlflow.db
    from dlkit.infrastructure.io import url_resolver

    db_path = (mlruns_dir(env=env) / "mlflow.db").resolve()
    return url_resolver.build_uri(db_path, scheme="sqlite")


def mlartifacts_dir(*, env: EnvironmentSettings | None = None) -> Path:
    return output("mlartifacts", env=env)


def optuna_storage_uri(*, env: EnvironmentSettings | None = None) -> str:
    from dlkit.infrastructure.io import url_resolver

    db_path = output("optuna.db", env=env)
    return url_resolver.build_uri(db_path, scheme="sqlite")
