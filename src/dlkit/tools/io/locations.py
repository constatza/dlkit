"""Centralized, environment-aware path policy (pure, no side effects).

All functions consult the current path override context first, then fall back to
the DLKitEnvironment root. Automatically detects test environment and routes
artifacts to tests/ directory. No directories are created here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dlkit.interfaces.api.overrides.path_context import (
    get_current_path_context,
    resolve_with_context,
)
from dlkit.tools.config.environment import DLKitEnvironment, env as global_environment


def _root_from_context() -> Path | None:
    try:
        ctx = get_current_path_context()
        if ctx and getattr(ctx, "root_dir", None):
            return Path(ctx.root_dir).resolve()
    except Exception:
        return None
    return None


def _is_test_environment() -> bool:
    """Detect if running in test environment.

    Returns:
        bool: True if running under pytest or test conditions
    """
    # Check for explicit test mode environment variable
    if os.environ.get("DLKIT_TEST_MODE"):
        return True

    # Check if pytest is running
    if "pytest" in sys.modules:
        return True

    # Check if invoked via pytest command
    if any("pytest" in arg for arg in sys.argv):
        return True

    # Check for pytest environment variables
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True

    return False


def _get_test_artifacts_root() -> Path:
    """Get the root directory for test artifacts.

    Returns:
        Path: tests/artifacts directory relative to project root
    """
    # Find project root (where tests/ directory exists)
    current = Path.cwd()

    # Try to find tests directory going up the directory tree
    for parent in [current] + list(current.parents):
        tests_dir = parent / "tests"
        if tests_dir.exists() and tests_dir.is_dir():
            return tests_dir / "artifacts"

    # Fallback: use current working directory + tests/artifacts
    return current / "tests" / "artifacts"


def root() -> Path:
    """Resolved root directory for DLKit operations.

    Priority:
    1) Thread-local path override context (set by CLI/API)
    2) Current working directory
    """
    ctx_root = _root_from_context()
    return ctx_root or Path.cwd()


def output(*parts: str, env: DLKitEnvironment | None = None) -> Path:
    """Resolve a path under the standard output directory.

    Honors path override context for output dir when present.
    In test environment, automatically routes to tests/artifacts.
    """
    # Check if we're in a test environment
    if _is_test_environment():
        # Use test-specific artifacts directory
        test_root = _get_test_artifacts_root()
        return (test_root.joinpath(*parts)).resolve()

    # Normal production path resolution
    base = resolve_with_context("output", env or global_environment)
    return (base.joinpath(*parts)).resolve()


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
    db_path = mlruns_dir(env=env) / "mlflow.db"
    # Always format as POSIX so URIs remain consistent across platforms
    return f"sqlite:///{db_path.as_posix()}"


def mlartifacts_dir(*, env: DLKitEnvironment | None = None) -> Path:
    return output("mlartifacts", env=env)


def optuna_storage_uri(*, env: DLKitEnvironment | None = None) -> str:
    db_path = output("optuna.db", env=env)
    # Always format as POSIX so URIs remain consistent across platforms
    return f"sqlite:///{db_path.as_posix()}"
