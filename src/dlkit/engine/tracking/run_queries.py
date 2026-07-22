"""Pure MLflow run lookup helpers, no artifact downloading.

Standalone, explicitly-invoked utilities that resolve a target run id for
run-based checkpoint selection: either the most recently started run in an
experiment, or the child runs of a parent run (e.g. a
``MultiRunOrchestrator`` sweep's Study run, or a parent tagged by an external
caller via ``mlflow.parentRunId``). Neither function downloads anything —
see ``checkpoint_recovery.py`` for the download step that follows a resolved
run id.
"""

from __future__ import annotations

from mlflow.exceptions import MlflowException

from dlkit.common.errors import WorkflowError

from .mlflow_client_factory import MLflowClientFactory


def find_latest_run_id(
    *,
    experiment_name: str,
    tracking_uri: str | None = None,
) -> str:
    """Find the most recently started active run in an experiment.

    Args:
        experiment_name: Name of the MLflow experiment to search.
        tracking_uri: Optional explicit MLflow tracking URI override.

    Returns:
        Run id of the active run with the latest ``start_time`` in the
        experiment.

    Raises:
        WorkflowError: If the experiment does not exist, or has zero active
            (non soft-deleted) runs.
    """
    from mlflow.entities import ViewType

    client = MLflowClientFactory.create_client(tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise WorkflowError(
            f"Experiment {experiment_name!r} not found.",
            {"experiment_name": experiment_name},
        )

    runs = client.search_runs(
        [experiment.experiment_id],
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise WorkflowError(
            f"Experiment {experiment_name!r} has no active runs.",
            {"experiment_name": experiment_name, "experiment_id": experiment.experiment_id},
        )
    return runs[0].info.run_id


def find_child_run_ids(
    *,
    parent_run_id: str,
    tracking_uri: str | None = None,
) -> tuple[str, ...]:
    """Find all active child runs of a parent run, in creation order.

    Matches on the ``mlflow.parentRunId`` tag rather than any dlkit-specific
    orchestration mechanism, so it finds children regardless of whether they
    were created via ``MLflowResourceManager.create_run(nested=True)`` or
    tagged directly by an external caller.

    Args:
        parent_run_id: MLflow run id of the parent run.
        tracking_uri: Optional explicit MLflow tracking URI override.

    Returns:
        Tuple of child run ids ordered by ascending ``start_time``.

    Raises:
        WorkflowError: If ``parent_run_id`` does not exist, or has zero
            active children (treated as a caller mistake, not a valid empty
            batch).
    """
    from mlflow.entities import ViewType

    client = MLflowClientFactory.create_client(tracking_uri)
    try:
        parent_run = client.get_run(parent_run_id)
    except MlflowException as exc:
        raise WorkflowError(
            f"Parent run {parent_run_id!r} not found: {exc}",
            {"parent_run_id": parent_run_id},
        ) from exc

    children = client.search_runs(
        [parent_run.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["attributes.start_time ASC"],
    )
    if not children:
        raise WorkflowError(
            f"Parent run {parent_run_id!r} has no active child runs.",
            {"parent_run_id": parent_run_id},
        )
    return tuple(run.info.run_id for run in children)
