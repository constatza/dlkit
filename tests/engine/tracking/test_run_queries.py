"""Tests for `run_queries.find_latest_run_id` and `find_child_run_ids`.

All tests run against a real (local sqlite) MLflow tracking backend via the
`mlflow_client`/`experiment_id`/`tracking_uri` fixtures in `conftest.py` —
no mocking of the MLflow client itself.
"""

from __future__ import annotations

import mlflow
import pytest
from mlflow import MlflowClient

from dlkit.common.errors import WorkflowError
from dlkit.engine.tracking.run_queries import find_child_run_ids, find_latest_run_id


def test_find_latest_run_id_returns_most_recently_started_active_run(
    mlflow_client: MlflowClient, experiment_id: str, experiment_name: str, tracking_uri: str
) -> None:
    mlflow_client.create_run(experiment_id, start_time=1_000, run_name="earliest")
    middle = mlflow_client.create_run(experiment_id, start_time=2_000, run_name="middle")
    latest = mlflow_client.create_run(experiment_id, start_time=3_000, run_name="latest")
    del middle

    result = find_latest_run_id(experiment_name=experiment_name, tracking_uri=tracking_uri)

    assert result == latest.info.run_id


def test_find_latest_run_id_excludes_soft_deleted_run(
    mlflow_client: MlflowClient, experiment_id: str, experiment_name: str, tracking_uri: str
) -> None:
    active = mlflow_client.create_run(experiment_id, start_time=1_000, run_name="active")
    deleted = mlflow_client.create_run(experiment_id, start_time=2_000, run_name="deleted")
    mlflow_client.delete_run(deleted.info.run_id)

    result = find_latest_run_id(experiment_name=experiment_name, tracking_uri=tracking_uri)

    assert result == active.info.run_id


def test_find_latest_run_id_raises_for_nonexistent_experiment(tracking_uri: str) -> None:
    with pytest.raises(WorkflowError, match="not found"):
        find_latest_run_id(experiment_name="does-not-exist", tracking_uri=tracking_uri)


def test_find_latest_run_id_raises_for_experiment_with_zero_active_runs(
    mlflow_client: MlflowClient, experiment_id: str, experiment_name: str, tracking_uri: str
) -> None:
    only_run = mlflow_client.create_run(experiment_id, start_time=1_000, run_name="only")
    mlflow_client.delete_run(only_run.info.run_id)

    with pytest.raises(WorkflowError, match="no active runs"):
        find_latest_run_id(experiment_name=experiment_name, tracking_uri=tracking_uri)


def test_find_child_run_ids_returns_children_in_creation_order_via_tags(
    mlflow_client: MlflowClient, experiment_id: str, tracking_uri: str
) -> None:
    parent = mlflow_client.create_run(experiment_id, start_time=1_000, run_name="parent")
    child_a = mlflow_client.create_run(experiment_id, start_time=2_000, run_name="child-a")
    child_b = mlflow_client.create_run(experiment_id, start_time=3_000, run_name="child-b")
    mlflow_client.set_tag(child_a.info.run_id, "mlflow.parentRunId", parent.info.run_id)
    mlflow_client.set_tag(child_b.info.run_id, "mlflow.parentRunId", parent.info.run_id)
    # A run in the same experiment that is not a child must be excluded.
    mlflow_client.create_run(experiment_id, start_time=4_000, run_name="unrelated")

    result = find_child_run_ids(parent_run_id=parent.info.run_id, tracking_uri=tracking_uri)

    assert result == (child_a.info.run_id, child_b.info.run_id)


def test_find_child_run_ids_excludes_soft_deleted_child(
    mlflow_client: MlflowClient, experiment_id: str, tracking_uri: str
) -> None:
    parent = mlflow_client.create_run(experiment_id, start_time=1_000, run_name="parent")
    active_child = mlflow_client.create_run(experiment_id, start_time=2_000, run_name="active")
    deleted_child = mlflow_client.create_run(experiment_id, start_time=3_000, run_name="deleted")
    mlflow_client.set_tag(active_child.info.run_id, "mlflow.parentRunId", parent.info.run_id)
    mlflow_client.set_tag(deleted_child.info.run_id, "mlflow.parentRunId", parent.info.run_id)
    mlflow_client.delete_run(deleted_child.info.run_id)

    result = find_child_run_ids(parent_run_id=parent.info.run_id, tracking_uri=tracking_uri)

    assert result == (active_child.info.run_id,)


def test_find_child_run_ids_raises_for_parent_with_no_children(
    mlflow_client: MlflowClient, experiment_id: str, tracking_uri: str
) -> None:
    parent = mlflow_client.create_run(experiment_id, start_time=1_000, run_name="lonely-parent")

    with pytest.raises(WorkflowError, match="no active child runs"):
        find_child_run_ids(parent_run_id=parent.info.run_id, tracking_uri=tracking_uri)


def test_find_child_run_ids_raises_for_nonexistent_parent(tracking_uri: str) -> None:
    with pytest.raises(WorkflowError, match="not found"):
        find_child_run_ids(parent_run_id="does-not-exist", tracking_uri=tracking_uri)


def test_find_child_run_ids_matches_nested_orchestration_path(
    mlflow_client: MlflowClient, tracking_uri: str
) -> None:
    """The query must not depend on dlkit's own `nested=True` orchestration path.

    A parent run tagged with two children purely via direct
    `client.set_tag(..., "mlflow.parentRunId", ...)` calls (mirroring how an
    external caller such as dl-experiments' `tag_run_parent` convention would
    tag runs, with no `MultiRunOrchestrator` involved) must be discovered
    identically to the `mlflow.start_run(nested=True)` case.
    """
    with mlflow.start_run(run_name="nested-parent", nested=False) as parent:
        with mlflow.start_run(run_name="nested-child-1", nested=True) as nested_child_1:
            pass
        with mlflow.start_run(run_name="nested-child-2", nested=True) as nested_child_2:
            pass

    nested_result = find_child_run_ids(parent_run_id=parent.info.run_id, tracking_uri=tracking_uri)
    assert set(nested_result) == {nested_child_1.info.run_id, nested_child_2.info.run_id}

    tagged_parent = mlflow_client.create_run(parent.info.experiment_id, run_name="tagged-parent")
    tagged_child_1 = mlflow_client.create_run(parent.info.experiment_id, run_name="tagged-child-1")
    tagged_child_2 = mlflow_client.create_run(parent.info.experiment_id, run_name="tagged-child-2")
    mlflow_client.set_tag(
        tagged_child_1.info.run_id, "mlflow.parentRunId", tagged_parent.info.run_id
    )
    mlflow_client.set_tag(
        tagged_child_2.info.run_id, "mlflow.parentRunId", tagged_parent.info.run_id
    )

    tagged_result = find_child_run_ids(
        parent_run_id=tagged_parent.info.run_id, tracking_uri=tracking_uri
    )
    assert set(tagged_result) == {tagged_child_1.info.run_id, tagged_child_2.info.run_id}
