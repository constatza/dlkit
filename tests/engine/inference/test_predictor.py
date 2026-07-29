"""Tests for ``CheckpointPredictor``'s forward-kwarg contract validation.

Covers the bug this investigation set out to fix: a caller passing the
wrong keyword argument name (e.g. ``x=`` for a model that expects
``branch=``) must get a dlkit-authored ``ForwardContractError`` naming the
expected kwargs, not a raw ``TypeError`` from inside the model call.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dlkit.common.errors import ForwardContractError
from dlkit.engine.inference.config import PredictorConfig
from dlkit.engine.inference.predictor import CheckpointPredictor


@pytest.fixture
def checkpoint_with_forward_arg_map(tmp_path: Path) -> Path:
    """A ``torch.nn.Linear`` checkpoint with a persisted named forward-arg contract.

    Mirrors what a real dlkit training run persists into ``dlkit_metadata``
    for a single named-kwarg model input, without going through the full
    Lightning wrapper/checkpoint-serializer machinery.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        Path: Location of the saved checkpoint file.
    """
    model = torch.nn.Linear(10, 5).to(torch.float32)
    model.eval()

    checkpoint_path = tmp_path / "model_with_contract.ckpt"
    checkpoint = {
        "state_dict": model.state_dict(),
        "dlkit_metadata": {
            "model_settings": {
                "name": "Linear",
                "module_path": "torch.nn",
                "hyper_kwargs": {"in_features": 10, "out_features": 5},
            },
            "feature_names": ["input"],
            "forward_arg_map": {"input": "input"},
        },
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def loaded_predictor(checkpoint_with_forward_arg_map: Path) -> CheckpointPredictor:
    """A loaded predictor backed by ``checkpoint_with_forward_arg_map``."""
    config = PredictorConfig(checkpoint_path=checkpoint_with_forward_arg_map, device="cpu")
    return CheckpointPredictor(config)


@pytest.fixture
def loaded_legacy_predictor(simple_checkpoint: Path) -> CheckpointPredictor:
    """A loaded predictor backed by a checkpoint with no persisted forward_arg_map."""
    config = PredictorConfig(checkpoint_path=simple_checkpoint, device="cpu")
    return CheckpointPredictor(config)


class TestDescribeInputs:
    def test_returns_persisted_contract(self, loaded_predictor: CheckpointPredictor) -> None:
        """describe_inputs() surfaces the checkpoint's named forward-arg contract."""
        assert loaded_predictor.describe_inputs() == {"input": "input"}

    def test_empty_for_legacy_checkpoint(
        self, loaded_legacy_predictor: CheckpointPredictor
    ) -> None:
        """A checkpoint with no persisted forward_arg_map describes no named contract."""
        assert loaded_legacy_predictor.describe_inputs() == {}


class TestPredictKwargValidation:
    def test_correct_kwarg_name_succeeds(self, loaded_predictor: CheckpointPredictor) -> None:
        """The exact contract name from describe_inputs() is accepted."""
        output = loaded_predictor.predict(input=torch.randn(2, 10))
        assert output.predictions.shape == (2, 5)

    def test_wrong_kwarg_name_raises_forward_contract_error(
        self, loaded_predictor: CheckpointPredictor
    ) -> None:
        """A wrong kwarg name raises a dlkit-authored error naming the expected kwargs,
        not a raw TypeError from inside the model call — the original bug report."""
        with pytest.raises(ForwardContractError, match=r"\['input'\]") as exc_info:
            loaded_predictor.predict(x=torch.randn(2, 10))
        assert "x" in str(exc_info.value)

    def test_legacy_checkpoint_skips_validation(
        self, loaded_legacy_predictor: CheckpointPredictor
    ) -> None:
        """No persisted forward_arg_map means no named contract to validate against —
        positional dispatch is unaffected."""
        output = loaded_legacy_predictor.predict(torch.randn(2, 10))
        assert output.predictions.shape == (2, 5)
