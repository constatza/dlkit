"""Centralized, environment-aware path policy (pure, no side effects).

All functions consult the current path override context first, then fall back to
the DLKitEnvironment root. No directories are created here.
"""

from __future__ import annotations

import os
from pathlib import Path

from dlkit.tools.config.environment import DLKitEnvironment
from dlkit.tools.config.environment import env as global_environment
from dlkit.tools.io.path_context import get_current_path_context, resolve_with_context


def _root_from_context() -> Path | None:
    try:
        ctx = get_current_path_context()
        root_dir = getattr(ctx, "root_dir", None) if ctx else None
        if root_dir is not None:
            return Path(root_dir).resolve()
    except Exception:
        return None
    return None


def root() -> Path:
    """Resolved root directory for DLKit operations.

    Priority:
    1) Thread-local path override context (set by CLI/API)
    2) DLKitEnvironment.root_dir (from DLKIT_ROOT_DIR env var or SESSION.root_dir)
    3) Current working directory
    """
    from loguru import logger

    ctx_root = _root_from_context()
    if ctx_root:
        logger.debug(f"Using root from PathOverrideContext: {ctx_root}")
        return ctx_root

    # Check DLKitEnvironment fallback
    try:
        env_root = global_environment.get_root_path()
        if env_root != Path.cwd():
            logger.debug(f"Using root from DLKitEnvironment: {env_root}")
            return env_root
    except Exception:
        pass

    # Fallback to CWD
    cwd = Path.cwd()
    logger.debug(f"Using root from current working directory (fallback): {cwd}")
    return cwd


def output(*parts: str, env: DLKitEnvironment | None = None) -> Path:
    """Resolve a path under the standard output directory.

    Honors path override context for output dir when present.
    """
    from loguru import logger

    base = resolve_with_context("output", env or global_environment)
    result = (base.joinpath(*parts)).resolve()

    # Warn if we're falling back to CWD when SESSION.root_dir might be expected
    cwd = Path.cwd()
    if base.resolve() == (cwd / "output").resolve():
        if not os.environ.get("DLKIT_ROOT_DIR"):
            ctx = get_current_path_context()
            if not ctx or not ctx.root_dir:
                logger.debug(
                    f"Output path resolved to CWD/output: {result}. "
                    "If SESSION.root_dir is set, ensure PathOverrideContext is active."
                )

    return result


def predictions_dir(*, env: DLKitEnvironment | None = None) -> Path:
    return output("predictions", env=env)


def checkpoints_dir(*, env: DLKitEnvironment | None = None) -> Path:
    return output("checkpoints", env=env)


def splits_dir(*, env: DLKitEnvironment | None = None) -> Path:
    return output("splits", env=env)


def figures_dir(*, env: DLKitEnvironment | None = None) -> Path:
    return output("figures", env=env)


def lightning_work_dir(*, env: DLKitEnvironment | None = None) -> Path:
    return output("lightning", env=env)


def mlruns_dir(*, env: DLKitEnvironment | None = None) -> Path:
    return output("mlruns", env=env)


def mlruns_backend_uri(*, env: DLKitEnvironment | None = None) -> str:
    # sqlite file under output/mlruns/mlflow.db
    from dlkit.tools.io import url_resolver

    db_path = (mlruns_dir(env=env) / "mlflow.db").resolve()
    return url_resolver.build_uri(db_path, scheme="sqlite")


def mlartifacts_dir(*, env: DLKitEnvironment | None = None) -> Path:
    return output("mlartifacts", env=env)


def optuna_storage_uri(*, env: DLKitEnvironment | None = None) -> str:
    from dlkit.tools.io import url_resolver

    db_path = output("optuna.db", env=env)
    return url_resolver.build_uri(db_path, scheme="sqlite")
