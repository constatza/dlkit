"""Centralized path helpers for DLKit-owned internal files."""

from __future__ import annotations

from pathlib import Path

from dlkit.infrastructure.config.environment import EnvironmentSettings


def root() -> Path:
    """Return the process working directory used for runtime-relative operations."""
    return Path.cwd().resolve()


def output(*parts: str, env: EnvironmentSettings | None = None) -> Path:
    """Resolve a path under DLKit's internal workspace directory."""
    settings = env or EnvironmentSettings()
    return (settings.get_internal_dir_path().joinpath(*parts)).resolve()


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
    return output("mlflow", env=env)


def mlruns_backend_uri(*, env: EnvironmentSettings | None = None) -> str:
    # sqlite file under .dlkit/mlflow/mlflow.db
    from dlkit.infrastructure.io import url_resolver

    db_path = (mlruns_dir(env=env) / "mlflow.db").resolve()
    return url_resolver.build_uri(db_path, scheme="sqlite")


def mlartifacts_dir(*, env: EnvironmentSettings | None = None) -> Path:
    return output("mlflow", "artifacts", env=env)


def optuna_storage_uri(*, env: EnvironmentSettings | None = None) -> str:
    from dlkit.infrastructure.io import url_resolver

    db_path = output("optuna.db", env=env)
    return url_resolver.build_uri(db_path, scheme="sqlite")
