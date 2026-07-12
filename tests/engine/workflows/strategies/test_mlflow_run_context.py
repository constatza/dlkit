"""Tests for ClientBasedRunContext model operations."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mlflow
import numpy as np
import pytest
import torch
from sklearn.linear_model import LinearRegression

from dlkit.engine.tracking.artifact_logger import _build_pt2_signature
from dlkit.engine.tracking.mlflow_run_context import ClientBasedRunContext


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
    assert "serialization_format" not in mocked_log_model.call_args.kwargs


def test_log_model_forwards_pt2_serialization_for_torch_modules() -> None:
    context = ClientBasedRunContext(client=Mock(), run_id="run-1", tracking_uri="sqlite:///test.db")

    with patch("mlflow.pytorch.log_model") as mocked_log_model:
        uri = context.log_model(
            model=torch.nn.Linear(2, 1),
            artifact_path="model",
            model_serialization_format="pt2",
        )

    assert uri == "runs:/run-1/model"
    mocked_log_model.assert_called_once()
    assert mocked_log_model.call_args.kwargs["serialization_format"] == "pt2"


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
    assert "serialization_format" not in mocked_log_model.call_args.kwargs


def test_log_model_ignores_serialization_format_for_sklearn_estimators() -> None:
    context = ClientBasedRunContext(client=Mock(), run_id="run-2", tracking_uri="sqlite:///test.db")

    with patch("mlflow.sklearn.log_model") as mocked_log_model:
        uri = context.log_model(
            model=LinearRegression(),
            artifact_path="model",
            model_serialization_format="pt2",
        )

    assert uri == "runs:/run-2/model"
    mocked_log_model.assert_called_once()
    assert "serialization_format" not in mocked_log_model.call_args.kwargs


def test_log_model_pt2_can_be_loaded_for_inference(tmp_path: Path) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = mlflow.create_experiment(
        "pt2-load-test",
        artifact_location=(tmp_path / "artifacts").as_uri(),
    )

    model = torch.nn.Linear(4, 2)
    object.__setattr__(
        model,
        "_checkpoint_metadata",
        SimpleNamespace(context=SimpleNamespace(input_shapes={"x": (4,)})),
    )
    input_example = np.zeros((1, 4), dtype=np.float32)

    with mlflow.start_run(experiment_id=experiment_id) as run:
        context = ClientBasedRunContext(
            client=mlflow.MlflowClient(),
            run_id=run.info.run_id,
            tracking_uri=tracking_uri,
            experiment_id=experiment_id,
        )
        model_uri = context.log_model(
            model=model,
            artifact_path="model",
            input_example=input_example,
            signature=_build_pt2_signature(model),
            model_serialization_format="pt2",
        )

    assert model_uri is not None
    loaded = mlflow.pytorch.load_model(model_uri)
    result = loaded(torch.zeros(1, 4))

    assert tuple(result.shape) == (1, 2)


def test_log_model_pt2_single_input_pyfunc_predict_does_not_warn_or_fail(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Regression: a named single-input signature broke MLflow pyfunc serving.

    MLflow's pytorch flavor wraps a named-schema example into a dict at
    predict time and rejects it, which previously showed up as a "Failed to
    validate serving input example" warning during log_model and a real
    TypeError from mlflow.pyfunc.load_model(...).predict(...).
    """
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = mlflow.create_experiment(
        "pt2-pyfunc-test",
        artifact_location=(tmp_path / "artifacts").as_uri(),
    )

    model = torch.nn.Linear(4, 2)
    object.__setattr__(
        model,
        "_checkpoint_metadata",
        SimpleNamespace(context=SimpleNamespace(input_shapes={"x": (4,)})),
    )
    input_example = np.zeros((1, 4), dtype=np.float32)

    with (
        caplog.at_level(logging.WARNING, logger="mlflow.models.model"),
        mlflow.start_run(experiment_id=experiment_id) as run,
    ):
        context = ClientBasedRunContext(
            client=mlflow.MlflowClient(),
            run_id=run.info.run_id,
            tracking_uri=tracking_uri,
            experiment_id=experiment_id,
        )
        model_uri = context.log_model(
            model=model,
            artifact_path="model",
            input_example=input_example,
            signature=_build_pt2_signature(model),
            model_serialization_format="pt2",
        )

    assert "Failed to validate serving input example" not in caplog.text
    assert model_uri is not None
    result = mlflow.pyfunc.load_model(model_uri).predict(input_example)

    assert tuple(np.asarray(result).shape) == (1, 2)


def test_log_model_multi_input_pyfunc_predict_is_unsupported(tmp_path: Path) -> None:
    """Documents the known upstream limitation: pyfunc serving cannot handle
    multi-tensor inputs under the pytorch flavor, regardless of signature naming.

    This is why ArtifactLogger._log_model_artifact logs an explicit warning for
    multi-input models instead of relying on MLflow's own (silent) failure.
    """
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = mlflow.create_experiment(
        "multi-input-pyfunc-test",
        artifact_location=(tmp_path / "artifacts").as_uri(),
    )

    model = torch.nn.Linear(4, 2)
    object.__setattr__(
        model,
        "_checkpoint_metadata",
        SimpleNamespace(context=SimpleNamespace(input_shapes={"x": (4,), "y": (8,)})),
    )
    input_example = (
        np.zeros((1, 4), dtype=np.float32),
        np.zeros((1, 8), dtype=np.float32),
    )

    with mlflow.start_run(experiment_id=experiment_id) as run:
        context = ClientBasedRunContext(
            client=mlflow.MlflowClient(),
            run_id=run.info.run_id,
            tracking_uri=tracking_uri,
            experiment_id=experiment_id,
        )
        model_uri = context.log_model(
            model=model,
            artifact_path="model",
            input_example=input_example,
            signature=_build_pt2_signature(model),
            model_serialization_format="pickle",
        )

    assert model_uri is not None
    with pytest.raises(Exception, match="Model is missing inputs"):
        mlflow.pyfunc.load_model(model_uri).predict(input_example)


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


def test_log_artifact_content_routes_bytes_through_log_binary_artifact() -> None:
    context = ClientBasedRunContext(client=Mock(), run_id="run-9", tracking_uri="sqlite:///test.db")

    with patch(
        "dlkit.engine.tracking.mlflow_run_context.log_binary_artifact"
    ) as mocked_log_binary_artifact:
        context.log_artifact_content(b"\x89PNG\r\n", "plots/loss_curve.png")

    mocked_log_binary_artifact.assert_called_once_with(
        context._client, "run-9", b"\x89PNG\r\n", "plots/loss_curve.png"
    )


def test_log_artifact_content_routes_str_through_log_text() -> None:
    client = Mock()
    context = ClientBasedRunContext(
        client=client, run_id="run-10", tracking_uri="sqlite:///test.db"
    )

    context.log_artifact_content("hello", "config/notes.txt")

    client.log_text.assert_called_once_with("run-10", "hello", "config/notes.txt")


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
