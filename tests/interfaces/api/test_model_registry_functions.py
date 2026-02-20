"""Tests for model registry API helpers."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from dlkit.interfaces.api.functions.model_registry import (
    build_registered_model_uri,
    get_model_version,
    list_model_versions,
    load_registered_model,
    register_logged_model,
    set_registered_model_alias,
    set_registered_model_version_tag,
    set_registered_model_version_tags,
    search_registered_models,
)


def test_build_registered_model_uri_from_version() -> None:
    assert build_registered_model_uri("MyModel", version=3) == "models:/MyModel/3"


def test_build_registered_model_uri_from_default_alias() -> None:
    assert build_registered_model_uri("MyModel") == "models:/MyModel@latest"


def test_build_registered_model_uri_rejects_version_and_alias_together() -> None:
    with pytest.raises(ValueError, match="either version or alias"):
        build_registered_model_uri("MyModel", version=1, alias="candidate")


@patch("mlflow.tracking.MlflowClient")
def test_search_registered_models_uses_name_filter(mock_client_cls: Mock) -> None:
    mock_client = Mock()
    mock_client.search_registered_models.return_value = ["model-a"]
    mock_client_cls.return_value = mock_client

    result = search_registered_models("ModelA", tracking_uri="http://localhost:5000")

    assert result == ["model-a"]
    mock_client.search_registered_models.assert_called_once_with(
        filter_string="name = 'ModelA'"
    )


@patch("mlflow.tracking.MlflowClient")
def test_list_model_versions_sorts_versions(mock_client_cls: Mock) -> None:
    mock_client = Mock()
    mock_client.search_model_versions.return_value = [
        Mock(version="5"),
        Mock(version="1"),
        Mock(version="3"),
    ]
    mock_client_cls.return_value = mock_client

    versions = list_model_versions("ModelA")

    assert versions == [1, 3, 5]


@patch("mlflow.tracking.MlflowClient")
def test_get_model_version_delegates_to_client(mock_client_cls: Mock) -> None:
    mock_client = Mock()
    expected = Mock(version="2")
    mock_client.get_model_version.return_value = expected
    mock_client_cls.return_value = mock_client

    result = get_model_version("ModelA", 2)

    assert result is expected
    mock_client.get_model_version.assert_called_once_with(name="ModelA", version="2")


@patch("mlflow.tracking.MlflowClient")
def test_register_logged_model_creates_version(mock_client_cls: Mock) -> None:
    mock_client = Mock()
    expected_version = Mock(version="1")
    mock_client.create_model_version.return_value = expected_version
    mock_client_cls.return_value = mock_client

    result = register_logged_model("ModelA", run_id="run-123", artifact_path="model")

    assert result is expected_version
    mock_client.create_registered_model.assert_called_once_with("ModelA")
    mock_client.create_model_version.assert_called_once_with(
        name="ModelA",
        source="runs:/run-123/model",
        run_id="run-123",
    )


@patch("mlflow.tracking.MlflowClient")
def test_register_logged_model_ignores_existing_registered_model(mock_client_cls: Mock) -> None:
    mock_client = Mock()
    mock_client.create_registered_model.side_effect = Exception("RESOURCE_ALREADY_EXISTS: already exists")
    mock_client.create_model_version.return_value = Mock(version="4")
    mock_client_cls.return_value = mock_client

    register_logged_model("ModelA", run_id="run-123")

    mock_client.create_model_version.assert_called_once()


@patch("mlflow.tracking.MlflowClient")
def test_set_registered_model_alias_delegates_to_client(mock_client_cls: Mock) -> None:
    mock_client = Mock()
    mock_client_cls.return_value = mock_client

    set_registered_model_alias("ModelA", alias="prod", version=8)

    mock_client.set_registered_model_alias.assert_called_once_with(
        name="ModelA",
        alias="prod",
        version="8",
    )


@patch("mlflow.tracking.MlflowClient")
def test_set_registered_model_version_tag_delegates_to_client(mock_client_cls: Mock) -> None:
    mock_client = Mock()
    mock_client_cls.return_value = mock_client

    set_registered_model_version_tag("ModelA", version=8, key="team", value="ml")

    mock_client.set_model_version_tag.assert_called_once_with(
        name="ModelA",
        version="8",
        key="team",
        value="ml",
    )


@patch("mlflow.tracking.MlflowClient")
def test_set_registered_model_version_tags_sets_all(mock_client_cls: Mock) -> None:
    mock_client = Mock()
    mock_client_cls.return_value = mock_client

    set_registered_model_version_tags("ModelA", version=8, tags={"team": "ml", "stage": "qa"})

    assert mock_client.set_model_version_tag.call_count == 2


@patch("mlflow.pyfunc.load_model")
@patch("mlflow.sklearn.load_model")
@patch("mlflow.pytorch.load_model")
@patch("mlflow.set_tracking_uri")
@patch("mlflow.get_tracking_uri")
def test_load_registered_model_uses_alias_and_restores_tracking_uri(
    mock_get_tracking_uri: Mock,
    mock_set_tracking_uri: Mock,
    mock_pytorch_load_model: Mock,
    mock_sklearn_load_model: Mock,
    mock_load_model: Mock,
) -> None:
    mock_get_tracking_uri.return_value = "http://old-tracking-uri"
    mock_pytorch_load_model.return_value = object()

    load_registered_model(
        "ModelA",
        alias="candidate",
        tracking_uri="http://new-tracking-uri",
    )

    mock_pytorch_load_model.assert_called_once_with("models:/ModelA@candidate")
    mock_sklearn_load_model.assert_not_called()
    mock_load_model.assert_not_called()
    assert mock_set_tracking_uri.call_args_list[0].args[0] == "http://new-tracking-uri"
    assert mock_set_tracking_uri.call_args_list[-1].args[0] == "http://old-tracking-uri"


@patch("mlflow.pyfunc.load_model")
@patch("mlflow.sklearn.load_model")
@patch("mlflow.pytorch.load_model")
def test_load_registered_model_auto_fallback_to_sklearn(
    mock_pytorch_load_model: Mock,
    mock_sklearn_load_model: Mock,
    mock_pyfunc_load_model: Mock,
) -> None:
    mock_pytorch_load_model.side_effect = RuntimeError("pytorch failed")
    mock_sklearn_load_model.return_value = object()

    load_registered_model("ModelA", alias="dataset_A_latest")

    mock_pytorch_load_model.assert_called_once_with("models:/ModelA@dataset_A_latest")
    mock_sklearn_load_model.assert_called_once_with("models:/ModelA@dataset_A_latest")
    mock_pyfunc_load_model.assert_not_called()


@patch("mlflow.sklearn.load_model")
@patch("mlflow.pytorch.load_model")
def test_load_registered_model_with_sklearn_strategy(
    mock_pytorch_load_model: Mock,
    mock_sklearn_load_model: Mock,
) -> None:
    mock_sklearn_load_model.return_value = object()

    load_registered_model("ModelA", alias="benchmark_high_precision", flavor="sklearn")

    mock_sklearn_load_model.assert_called_once_with("models:/ModelA@benchmark_high_precision")
    mock_pytorch_load_model.assert_not_called()


@patch("mlflow.pyfunc.load_model")
def test_load_registered_model_with_pyfunc_strategy(mock_pyfunc_load_model: Mock) -> None:
    mock_pyfunc_load_model.return_value = object()

    load_registered_model("ModelA", version=5, flavor="pyfunc")

    mock_pyfunc_load_model.assert_called_once_with("models:/ModelA/5")
