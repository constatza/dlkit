"""Shared MLflow-tracking settings helper for workflow entrypoints and orchestrators.

Lives in ``factories`` (not ``entrypoints``) so both the API-layer executor
adapter (``interfaces.api.adapters``, which depends on ``entrypoints``) and
the engine-owned multirun orchestrator (``engine.workflows.multi_run``, which
does *not* depend on ``entrypoints`` — see ``tach.toml``: ``entrypoints``
depends on the general ``engine.workflows`` bucket, not the reverse) can
depend on one implementation without an ``engine.workflows -> entrypoints``
edge.
"""

from __future__ import annotations

from .build_strategy import WorkflowSettings


def apply_mlflow_flag(settings: WorkflowSettings, mlflow: bool) -> WorkflowSettings:
    """Return settings with tracking backend ensured when mlflow=True.

    When ``mlflow=True`` and the config has no explicit tracking backend,
    patches the tracking section to use MLflow so the engine enables it.

    Args:
        settings: Workflow configuration settings.
        mlflow: Whether to ensure MLflow tracking is configured.

    Returns:
        Settings with tracking backend set to ``"mlflow"``, or original
        settings if ``mlflow=False`` or tracking is already configured.
    """
    if not mlflow:
        return settings
    tracking = getattr(settings, "tracking", None)
    if tracking is not None and getattr(tracking, "backend", None) not in (None, "none"):
        return settings
    return settings.patch({"tracking": {"backend": "mlflow"}})
