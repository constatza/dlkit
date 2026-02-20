"""Tests for logged run-model API helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from dlkit.interfaces.api.functions.model_logged import (
    build_logged_model_uri,
    load_logged_model,
    search_logged_models,
)


def test_build_logged_model_uri() -> None:
    assert build_logged_model_uri("abc123") == "runs:/abc123/model"


def test_build_logged_model_uri_normalizes_artifact_path() -> None:
    assert build_logged_model_uri("abc123", artifact_path="/model/") == "runs:/abc123/model"


def test_load_logged_model_rejects_ambiguous_inputs() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        load_logged_model(run_id="run-1", model_uri="runs:/run-1/model")


@patch("mlflow.pyfunc.load_model")
@patch("mlflow.set_tracking_uri")
@patch("mlflow.get_tracking_uri")
@patch("dlkit.interfaces.api.functions.model_logged.create_mlflow_client")
def test_load_logged_model_from_run_id_and_restores_tracking_uri(
    mock_create_client: Mock,
    mock_get_tracking_uri: Mock,
    mock_set_tracking_uri: Mock,
    mock_load_model: Mock,
) -> None:
    mock_client = Mock()
    mock_client.get_run.return_value = SimpleNamespace(
        data=SimpleNamespace(tags={"mlflow_logged_model_uri": "models:/m-7"})
    )
    mock_create_client.return_value = mock_client
    mock_get_tracking_uri.return_value = "http://old-tracking-uri"
    mock_load_model.return_value = object()

    load_logged_model(
        run_id="run-7",
        artifact_path="model",
        tracking_uri="http://new-tracking-uri",
    )

    mock_load_model.assert_called_once_with("models:/m-7")
    assert mock_set_tracking_uri.call_args_list[0].args[0] == "http://new-tracking-uri"
    assert mock_set_tracking_uri.call_args_list[-1].args[0] == "http://old-tracking-uri"


@patch("dlkit.interfaces.api.functions.model_logged.create_mlflow_client")
def test_search_logged_models_returns_filtered_run_records(
    mock_create_client: Mock,
) -> None:
    mock_client = Mock()
    mock_create_client.return_value = mock_client
    mock_client.get_experiment_by_name.return_value = SimpleNamespace(experiment_id="12")

    run = SimpleNamespace(
        info=SimpleNamespace(
            run_id="run-42",
            experiment_id="12",
            status="FINISHED",
            start_time=1000,
            end_time=1200,
        ),
        data=SimpleNamespace(
            tags={
                "mlflow.runName": "trial-42",
                "mlflow_model_class": "FancyNet",
                "mlflow_logged_model_artifact_path": "model",
                "mlflow_logged_model_uri": "models:/m-42",
                "stage": "dev",
            }
        ),
    )
    mock_client.search_runs.return_value = [run]

    records = search_logged_models(
        model_name="FancyNet",
        experiment_name="exp-a",
        tags={"stage": "dev"},
    )

    assert len(records) == 1
    record = records[0]
    assert record.run_id == "run-42"
    assert record.experiment_id == "12"
    assert record.model_class == "FancyNet"
    assert record.model_uri == "models:/m-42"
    assert record.tags["stage"] == "dev"

    mock_client.search_runs.assert_called_once()
    called_filter = mock_client.search_runs.call_args.kwargs["filter_string"]
    assert "tags.mlflow_model_class = 'FancyNet'" in called_filter
    assert "tags.mlflow_logged_model_artifact_path = 'model'" in called_filter
