"""Tests for ClientBasedRunContext model operations."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sklearn.linear_model import LinearRegression

from dlkit.runtime.tracking.mlflow_run_context import ClientBasedRunContext


def test_log_model_uses_pytorch_flavor_for_torch_modules() -> None:
    context = ClientBasedRunContext(client=Mock(), run_id="run-1", tracking_uri="sqlite:///test.db")

    with patch("mlflow.pytorch.log_model") as mocked_log_model:
        uri = context.log_model(
            model=torch.nn.Linear(2, 1),
            artifact_path="model",
            registered_model_name="Linear",
        )

    assert uri == "runs:/run-1/model"
    mocked_log_model.assert_called_once()
    assert mocked_log_model.call_args.kwargs["registered_model_name"] == "Linear"


def test_log_model_uses_sklearn_flavor_for_estimators() -> None:
    context = ClientBasedRunContext(client=Mock(), run_id="run-2", tracking_uri="sqlite:///test.db")

    with patch("mlflow.sklearn.log_model") as mocked_log_model:
        uri = context.log_model(
            model=LinearRegression(),
            artifact_path="model",
            registered_model_name="LinearRegression",
        )

    assert uri == "runs:/run-2/model"
    mocked_log_model.assert_called_once()
    assert mocked_log_model.call_args.kwargs["registered_model_name"] == "LinearRegression"


def test_log_model_prefers_mlflow_returned_model_uri() -> None:
    context = ClientBasedRunContext(client=Mock(), run_id="run-5", tracking_uri="sqlite:///test.db")

    with patch("mlflow.pytorch.log_model") as mocked_log_model:
        mocked_log_model.return_value = SimpleNamespace(model_uri="models:/m-12345")
        uri = context.log_model(
            model=torch.nn.Linear(2, 1),
            artifact_path="model",
            registered_model_name="Linear",
        )

    assert uri == "models:/m-12345"


def test_get_latest_model_version_returns_numeric_max() -> None:
    client = Mock()
    client.search_model_versions.return_value = [
        SimpleNamespace(version="2"),
        SimpleNamespace(version="11"),
        SimpleNamespace(version="7"),
    ]
    context = ClientBasedRunContext(client=client, run_id="run-3", tracking_uri="sqlite:///test.db")

    latest = context.get_latest_model_version("MyModel")

    assert latest == 11
    client.search_model_versions.assert_called_once_with("name='MyModel'")


def test_get_latest_model_version_can_filter_by_run_id() -> None:
    client = Mock()
    client.search_model_versions.return_value = [
        SimpleNamespace(version="2", run_id="run-3", source="s3://bucket/model"),
        SimpleNamespace(version="11", run_id="other-run", source="s3://bucket/model"),
        SimpleNamespace(version="7", run_id="run-3", source="s3://bucket/model"),
    ]
    context = ClientBasedRunContext(client=client, run_id="run-3", tracking_uri="sqlite:///test.db")

    latest = context.get_latest_model_version("MyModel", run_id="run-3")

    assert latest == 7


def test_get_latest_model_version_can_filter_by_artifact_path() -> None:
    client = Mock()
    client.search_model_versions.return_value = [
        SimpleNamespace(version="2", run_id="run-3", source="s3://bucket/other_artifact"),
        SimpleNamespace(version="4", run_id="run-3", source="s3://bucket/model"),
    ]
    context = ClientBasedRunContext(client=client, run_id="run-3", tracking_uri="sqlite:///test.db")

    latest = context.get_latest_model_version(
        "MyModel",
        run_id="run-3",
        artifact_path="model",
    )

    assert latest == 4


def test_set_model_alias_delegates_to_client() -> None:
    client = Mock()
    context = ClientBasedRunContext(client=client, run_id="run-4", tracking_uri="sqlite:///test.db")

    context.set_model_alias("MyModel", "latest", 5)

    client.set_registered_model_alias.assert_called_once_with(
        name="MyModel",
        alias="latest",
        version="5",
    )


def test_set_model_version_tag_delegates_to_client() -> None:
    client = Mock()
    context = ClientBasedRunContext(client=client, run_id="run-6", tracking_uri="sqlite:///test.db")

    context.set_model_version_tag("MyModel", 5, "dataset.name", "dataset_A")

    client.set_model_version_tag.assert_called_once_with(
        name="MyModel",
        version="5",
        key="dataset.name",
        value="dataset_A",
    )


def test_log_dataset_converts_to_mlflow_entity_and_adds_context_tag() -> None:
    client = Mock()
    context = ClientBasedRunContext(client=client, run_id="run-7", tracking_uri="sqlite:///test.db")

    dataset_entity = Mock()
    dataset = Mock()
    dataset._to_mlflow_entity.return_value = dataset_entity

    context.log_dataset(dataset, context="training", tags={"split": "train"})

    client.log_inputs.assert_called_once()
    call_kwargs = client.log_inputs.call_args.kwargs
    assert call_kwargs["datasets"]

    dataset_input = call_kwargs["datasets"][0]
    assert dataset_input.dataset is dataset_entity
    assert {tag.key: tag.value for tag in dataset_input.tags} == {
        "split": "train",
        "mlflow.data.context": "training",
    }
