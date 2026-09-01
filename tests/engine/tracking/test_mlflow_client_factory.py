"""Tests for `MLflowClientFactory.get_or_create_experiment`'s race handling."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, RESOURCE_ALREADY_EXISTS

from dlkit.common.errors import TrackingError
from dlkit.engine.tracking.mlflow_client_factory import MLflowClientFactory


@pytest.fixture
def racing_client() -> MagicMock:
    """A mock `MlflowClient` whose `create_experiment` loses a create race.

    `get_experiment_by_name` returns `None` on its first call (no existing
    experiment found), then a real-looking experiment object on any
    subsequent call (as if a concurrent creator won in between).
    """
    client = MagicMock(spec=MlflowClient)
    existing = MagicMock(experiment_id="racing-experiment-id")
    client.get_experiment_by_name.side_effect = [None, existing]
    client.create_experiment.side_effect = MlflowException(
        "Experiment already exists", error_code=RESOURCE_ALREADY_EXISTS
    )
    return client


def test_get_or_create_experiment_returns_existing_id_without_creating() -> None:
    client = MagicMock(spec=MlflowClient)
    client.get_experiment_by_name.return_value = MagicMock(experiment_id="existing-id")

    experiment_id = MLflowClientFactory.get_or_create_experiment(client, "my-experiment")

    assert experiment_id == "existing-id"
    client.create_experiment.assert_not_called()


def test_get_or_create_experiment_creates_when_missing() -> None:
    client = MagicMock(spec=MlflowClient)
    client.get_experiment_by_name.return_value = None
    client.create_experiment.return_value = "new-id"

    experiment_id = MLflowClientFactory.get_or_create_experiment(client, "my-experiment")

    assert experiment_id == "new-id"


def test_get_or_create_experiment_resolves_race_against_concurrent_creator(
    racing_client: MagicMock,
) -> None:
    """Losing the create race isn't a failure -- the winner's id is fetched instead."""
    experiment_id = MLflowClientFactory.get_or_create_experiment(racing_client, "my-experiment")

    assert experiment_id == "racing-experiment-id"
    assert racing_client.get_experiment_by_name.call_count == 2


def test_get_or_create_experiment_raises_tracking_error_when_race_unresolvable() -> None:
    """If the experiment still can't be found after losing the race, that's a
    genuinely unexpected state -- raise, don't return a bogus id.
    """
    client = MagicMock(spec=MlflowClient)
    client.get_experiment_by_name.side_effect = [None, None]
    client.create_experiment.side_effect = MlflowException(
        "Experiment already exists", error_code=RESOURCE_ALREADY_EXISTS
    )

    with pytest.raises(TrackingError):
        MLflowClientFactory.get_or_create_experiment(client, "my-experiment")


def test_get_or_create_experiment_reraises_unrelated_mlflow_exceptions() -> None:
    """Only a `RESOURCE_ALREADY_EXISTS` conflict is treated as a race -- any other
    MLflow failure propagates unchanged.
    """
    client = MagicMock(spec=MlflowClient)
    client.get_experiment_by_name.return_value = None
    client.create_experiment.side_effect = MlflowException(
        "Something else went wrong", error_code=INTERNAL_ERROR
    )

    with pytest.raises(MlflowException) as exc_info:
        MLflowClientFactory.get_or_create_experiment(client, "my-experiment")
    assert not isinstance(exc_info.value, TrackingError)
