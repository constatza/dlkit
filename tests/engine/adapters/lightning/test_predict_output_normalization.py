"""Tests for model output normalization utilities and predict_step integration.

Covers:
- _unpack_model_output: structural unpacking of raw forward() output
- _normalize_output: recursive Tensor/TensorDict normalization
- _batch_size_of: batch size inference from various container types
- _leaf_dtype / _leaf_device: leaf tensor inspection with empty-TensorDict guard
- sequence_to_tensordict: positional-sequence wrapper
- predict_step integration: enriched-batch interface, target cloning, latent sentinel
- NamedBatchTransformer.inverse_transform_predictions for Tensor and TensorDict
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from tensordict import TensorDict, TensorDictBase
from torch import Tensor

from dlkit.engine.adapters.lightning.base import (
    _MAX_NORMALIZE_DEPTH,
    _batch_size_of,
    _leaf_device,
    _leaf_dtype,
    _normalize_output,
    _unpack_model_output,
)
from dlkit.engine.adapters.lightning.transform_pipeline import NamedBatchTransformer
from dlkit.infrastructure.utils.tensordict_utils import sequence_to_tensordict


def _expect_tensor(value: object) -> Tensor:
    assert isinstance(value, Tensor)
    return value


def _expect_tensordict(value: object) -> TensorDictBase:
    assert isinstance(value, TensorDictBase)
    return value


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bs() -> int:
    """Standard batch size used across tests."""
    return 4


@pytest.fixture
def t1(bs: int) -> Tensor:
    """First test tensor: (bs, 2)."""
    return torch.randn(bs, 2)


@pytest.fixture
def t2(bs: int) -> Tensor:
    """Second test tensor: (bs, 3)."""
    return torch.randn(bs, 3)


@pytest.fixture
def t3(bs: int) -> Tensor:
    """Third test tensor: (bs, 5)."""
    return torch.randn(bs, 5)


@pytest.fixture
def td(bs: int, t1: Tensor, t2: Tensor) -> TensorDict:
    """TensorDict with keys 'a' and 'b'."""
    return TensorDict({"a": t1, "b": t2}, batch_size=[bs])


# ---------------------------------------------------------------------------
# TestUnpackModelOutput
# ---------------------------------------------------------------------------


class TestUnpackModelOutput:
    def test_tensor_returns_tensor_and_none(self, t1: Tensor) -> None:
        preds, latents = _unpack_model_output(t1)
        assert preds is t1
        assert latents is None

    def test_tensordict_no_predictions_key(self, td: TensorDict) -> None:
        preds, latents = _unpack_model_output(td)
        assert preds is td
        assert latents is None

    def test_dict_without_predictions_key(self, t1: Tensor, t2: Tensor) -> None:
        d = {"out": t1, "z": t2}
        preds, latents = _unpack_model_output(d)
        assert preds is d
        assert latents is None

    def test_dict_with_predictions_key_only(self, t1: Tensor, t2: Tensor) -> None:
        d = {"predictions": t1, "other": t2}
        preds, latents = _unpack_model_output(d)
        assert preds is t1
        assert latents is None

    def test_dict_with_predictions_and_latents(self, t1: Tensor, t2: Tensor) -> None:
        d = {"predictions": t1, "latents": t2}
        preds, latents = _unpack_model_output(d)
        assert preds is t1
        assert latents is t2

    def test_tensordict_with_predictions_key(self, bs: int, t1: Tensor, t2: Tensor) -> None:
        raw = TensorDict({"predictions": t1, "latents": t2}, batch_size=[bs])
        preds, latents = _unpack_model_output(raw)
        assert preds is t1
        assert latents is t2

    def test_tensordict_with_predictions_no_latents(self, bs: int, t1: Tensor) -> None:
        raw = TensorDict({"predictions": t1}, batch_size=[bs])
        preds, latents = _unpack_model_output(raw)
        assert preds is t1
        assert latents is None

    def test_1_tuple(self, t1: Tensor) -> None:
        preds, latents = _unpack_model_output((t1,))
        assert preds is t1
        assert latents is None

    def test_2_tuple(self, t1: Tensor, t2: Tensor) -> None:
        preds, latents = _unpack_model_output((t1, t2))
        assert preds is t1
        assert latents is t2

    def test_3_tuple_packs_remainder_as_latents(self, t1: Tensor, t2: Tensor, t3: Tensor) -> None:
        preds, latents = _unpack_model_output((t1, t2, t3))
        assert preds is t1
        assert latents == (t2, t3)

    def test_list_raises_type_error(self, t1: Tensor) -> None:
        with pytest.raises(TypeError, match="ambiguous at top level"):
            _unpack_model_output([t1])

    def test_empty_tuple_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="empty tuple"):
            _unpack_model_output(())


# ---------------------------------------------------------------------------
# TestBatchSizeOf
# ---------------------------------------------------------------------------


class TestBatchSizeOf:
    def test_tensor(self, bs: int, t1: Tensor) -> None:
        assert _batch_size_of(t1) == bs

    def test_tensordict(self, bs: int, td: TensorDict) -> None:
        assert _batch_size_of(td) == bs

    def test_dict_with_tensor_value(self, bs: int, t1: Tensor) -> None:
        assert _batch_size_of({"x": t1}) == bs

    def test_dict_with_tensordict_value(self, bs: int, td: TensorDict) -> None:
        assert _batch_size_of({"nested": td}) == bs

    def test_empty_dict_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="empty dict"):
            _batch_size_of({})

    def test_unsupported_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="Cannot determine batch size"):
            _batch_size_of(42)


# ---------------------------------------------------------------------------
# TestNormalizeOutput
# ---------------------------------------------------------------------------


class TestNormalizeOutput:
    def test_tensor_passes_through(self, t1: Tensor, bs: int) -> None:
        result = _normalize_output(t1, "predictions", bs)
        assert result is t1

    def test_tensordict_passes_through(self, td: TensorDict, bs: int) -> None:
        result = _normalize_output(td, "predictions", bs)
        assert result is td

    def test_flat_dict_becomes_tensordict(self, t1: Tensor, t2: Tensor, bs: int) -> None:
        d = {"a": t1, "b": t2}
        result = _normalize_output(d, "predictions", bs)
        assert isinstance(result, TensorDict)
        assert int(result.batch_size[0]) == bs
        assert torch.equal(_expect_tensor(result["a"]), t1)
        assert torch.equal(_expect_tensor(result["b"]), t2)

    def test_dict_with_list_value_becomes_nested_positional_td(
        self, t1: Tensor, t2: Tensor, bs: int
    ) -> None:
        d = {"out": [t1, t2]}
        result = _normalize_output(d, "predictions", bs)
        assert isinstance(result, TensorDict)
        inner = _expect_tensordict(result["out"])
        assert torch.equal(_expect_tensor(inner["0"]), t1)
        assert torch.equal(_expect_tensor(inner["1"]), t2)

    def test_dict_with_nested_dict(self, t1: Tensor, bs: int) -> None:
        d = {"z": {"x": t1}}
        result = _normalize_output(d, "predictions", bs)
        assert isinstance(result, TensorDict)
        inner = _expect_tensordict(result["z"])
        assert torch.equal(_expect_tensor(inner["x"]), t1)

    def test_combined_nested_structure(self, t1: Tensor, t2: Tensor, t3: Tensor, bs: int) -> None:
        d = {"out": [t1, t2], "z": {"x": t3}}
        result = _normalize_output(d, "predictions", bs)
        assert isinstance(result, TensorDict)
        out_td = _expect_tensordict(result["out"])
        z_td = _expect_tensordict(result["z"])
        assert torch.equal(_expect_tensor(out_td["0"]), t1)
        assert torch.equal(_expect_tensor(out_td["1"]), t2)
        assert torch.equal(_expect_tensor(z_td["x"]), t3)

    def test_list_becomes_positional_td(self, t1: Tensor, t2: Tensor, bs: int) -> None:
        result = _normalize_output([t1, t2], "latents", bs)
        assert isinstance(result, TensorDict)
        assert torch.equal(_expect_tensor(result["0"]), t1)
        assert torch.equal(_expect_tensor(result["1"]), t2)

    def test_tuple_becomes_positional_td(self, t1: Tensor, t2: Tensor, bs: int) -> None:
        result = _normalize_output((t1, t2), "latents", bs)
        assert isinstance(result, TensorDict)
        assert torch.equal(_expect_tensor(result["0"]), t1)
        assert torch.equal(_expect_tensor(result["1"]), t2)

    def test_empty_dict_raises_value_error(self, bs: int) -> None:
        with pytest.raises(ValueError, match="empty dict"):
            _normalize_output({}, "predictions", bs)

    def test_empty_list_raises_value_error(self, bs: int) -> None:
        with pytest.raises(ValueError, match="empty sequence"):
            _normalize_output([], "latents", bs)

    def test_empty_tuple_raises_value_error(self, bs: int) -> None:
        with pytest.raises(ValueError, match="empty sequence"):
            _normalize_output((), "latents", bs)

    def test_unsupported_type_raises_type_error(self, bs: int) -> None:
        with pytest.raises(TypeError, match="unsupported output type"):
            _normalize_output(3.14, "predictions", bs)

    def test_depth_guard_prevents_infinite_recursion(self, t1: Tensor, bs: int) -> None:
        deep: Any = t1
        for _ in range(_MAX_NORMALIZE_DEPTH + 2):
            deep = {"inner": deep}
        with pytest.raises(RecursionError, match="nesting exceeds maximum depth"):
            _normalize_output(deep, "predictions", bs)


# ---------------------------------------------------------------------------
# TestLeafDtypeDevice
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_td() -> TensorDict:
    """An empty TensorDict with no leaves."""
    return TensorDict({}, batch_size=[4])


class TestLeafDtypeDevice:
    def test_leaf_dtype_tensor(self, t1: Tensor) -> None:
        assert _leaf_dtype(t1) == t1.dtype

    def test_leaf_dtype_tensordict(self, bs: int, t1: Tensor) -> None:
        td = TensorDict({"a": t1.to(torch.float64)}, batch_size=[bs])
        assert _leaf_dtype(td) == torch.float64

    def test_leaf_dtype_empty_tensordict_returns_float32(self, empty_td: TensorDict) -> None:
        """Empty TensorDict falls back to float32 (Bug 2 fix)."""
        assert _leaf_dtype(empty_td) == torch.float32

    def test_leaf_device_tensor(self, t1: Tensor) -> None:
        assert _leaf_device(t1) == t1.device

    def test_leaf_device_tensordict(self, bs: int, t1: Tensor) -> None:
        td = TensorDict({"a": t1}, batch_size=[bs])
        assert _leaf_device(td) == t1.device

    def test_leaf_device_empty_tensordict_returns_cpu(self, empty_td: TensorDict) -> None:
        """Empty TensorDict falls back to cpu (Bug 2 fix)."""
        assert _leaf_device(empty_td) == torch.device("cpu")


# ---------------------------------------------------------------------------
# TestSequenceToTensordict
# ---------------------------------------------------------------------------


class TestSequenceToTensordict:
    def test_two_tensors(self, t1: Tensor, t2: Tensor, bs: int) -> None:
        result = sequence_to_tensordict([t1, t2])
        assert isinstance(result, TensorDict)
        assert int(result.batch_size[0]) == bs
        assert torch.equal(_expect_tensor(result["0"]), t1)
        assert torch.equal(_expect_tensor(result["1"]), t2)

    def test_single_tensor(self, t1: Tensor, bs: int) -> None:
        result = sequence_to_tensordict([t1])
        assert torch.equal(_expect_tensor(result["0"]), t1)

    def test_tensordict_elements(self, td: TensorDict, bs: int) -> None:
        result = sequence_to_tensordict([td])
        assert result["0"] is td

    def test_empty_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="empty sequence"):
            sequence_to_tensordict([])


# ---------------------------------------------------------------------------
# Helpers for predict_step integration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def batch(bs: int) -> TensorDict:
    """Minimal TensorDict batch for predict_step tests."""
    return TensorDict(
        {
            "features": TensorDict({"x": torch.randn(bs, 2)}, batch_size=[bs]),
            "targets": TensorDict({"y": torch.randn(bs, 1)}, batch_size=[bs]),
        },
        batch_size=[bs],
    )


def _make_wrapper(enriched_batch: TensorDict, bs: int) -> Any:
    """Build a ProcessingLightningWrapper with a fixed invoker.

    The invoker returns *enriched_batch* directly, bypassing model invocation.
    This lets predict_step tests focus on the pipeline around the invoker.

    Args:
        enriched_batch: Pre-built TensorDict that the invoker will return.
            Must contain at least ``"predictions"``; optionally ``"latents"``.
        bs: Batch size (used to build the minimal model).

    Returns:
        Configured wrapper instance.
    """
    from torch import nn

    from dlkit.engine.adapters.lightning.base import ProcessingLightningWrapper
    from dlkit.engine.adapters.lightning.prediction_strategies import (
        DiscriminativePredictionStrategy,
    )
    from dlkit.engine.adapters.lightning.transform_pipeline import NamedBatchTransformer

    class _FixedInvoker:
        def invoke(self, model: nn.Module, batch: TensorDict) -> TensorDict:
            return enriched_batch

    class _MinimalWrapper(ProcessingLightningWrapper):
        def forward(self, *args: Any, **kwargs: Any) -> Tensor:
            return torch.zeros(bs, 1)

        def _run_step(
            self, batch: Any, batch_idx: int, stage: str
        ) -> tuple[Tensor, int | None, Any]:
            from dlkit.engine.adapters.lightning.base import _batch_size_of

            batch = self._model_invoker.invoke(self.model, batch)
            predictions = _expect_tensor(batch["predictions"])
            loss = self._loss_computer.compute(predictions, batch)
            batch_size = _batch_size_of(batch["predictions"])
            return loss, batch_size, batch

    loss_fn = torch.nn.MSELoss()

    class _LossComputer:
        def compute(self, predictions: Any, batch: Any) -> Tensor:
            return loss_fn(predictions, predictions)

    class _MetricsUpdater:
        def update(self, *a: Any, **kw: Any) -> None: ...
        def compute(self, stage: str) -> dict[str, Any]:
            return {}

        def reset(self, stage: str) -> None: ...

    optimizer_settings = MagicMock()
    optimizer_settings.lr = 1e-3

    batch_transformer = NamedBatchTransformer({}, {})
    invoker = _FixedInvoker()
    prediction_strategy = DiscriminativePredictionStrategy(
        model_invoker=invoker,
        batch_transformer=batch_transformer,
        predict_target_key="y",
    )

    return _MinimalWrapper(
        model=nn.Linear(2, 1),
        model_invoker=invoker,
        loss_computer=_LossComputer(),
        metrics_updater=_MetricsUpdater(),
        batch_transformer=batch_transformer,
        optimizer_settings=optimizer_settings,
        predict_target_key="y",
        prediction_strategy=prediction_strategy,
    )


# ---------------------------------------------------------------------------
# TestPredictStepOutput — enriched-batch interface
# ---------------------------------------------------------------------------


class TestPredictStepOutput:
    """predict_step reads 'predictions' and optional 'latents' from enriched batch."""

    def test_predictions_tensor_forwarded(self, bs: int, t1: Tensor, batch: TensorDict) -> None:
        """Plain Tensor in 'predictions' is forwarded as-is."""
        enriched = TensorDict({"predictions": t1}, batch_size=[bs])
        wrapper = _make_wrapper(enriched, bs)
        out = wrapper.predict_step(batch, 0)
        assert isinstance(out, TensorDict)
        assert torch.equal(_expect_tensor(out["predictions"]), t1)
        assert out["latents"].shape == (bs, 0)

    def test_predictions_tensordict_forwarded(
        self, bs: int, td: TensorDict, batch: TensorDict
    ) -> None:
        """TensorDict in 'predictions' is forwarded as-is (multi-head case)."""
        enriched = TensorDict({"predictions": td}, batch_size=[bs])
        wrapper = _make_wrapper(enriched, bs)
        out = wrapper.predict_step(batch, 0)
        assert isinstance(out["predictions"], TensorDict)
        assert out["latents"].shape == (bs, 0)

    def test_latents_sentinel_when_absent(self, bs: int, t1: Tensor, batch: TensorDict) -> None:
        """When enriched batch has no 'latents' key, output latents is (B, 0) sentinel."""
        enriched = TensorDict({"predictions": t1}, batch_size=[bs])
        wrapper = _make_wrapper(enriched, bs)
        out = wrapper.predict_step(batch, 0)
        assert out["latents"].shape == (bs, 0)

    def test_latents_from_enriched_batch(
        self, bs: int, t1: Tensor, t2: Tensor, batch: TensorDict
    ) -> None:
        """When enriched batch has 'latents', it is forwarded to output."""
        enriched = TensorDict({"predictions": t1, "latents": t2}, batch_size=[bs])
        wrapper = _make_wrapper(enriched, bs)
        out = wrapper.predict_step(batch, 0)
        assert torch.equal(_expect_tensor(out["latents"]), t2)

    def test_named_latents_tensordict_forwarded(
        self, bs: int, t1: Tensor, t2: Tensor, t3: Tensor, batch: TensorDict
    ) -> None:
        """Named latents (nested TensorDict) are forwarded from enriched batch."""
        latents_td = TensorDict({"mu": t2, "logvar": t3}, batch_size=[bs])
        enriched = TensorDict({"predictions": t1, "latents": latents_td}, batch_size=[bs])
        wrapper = _make_wrapper(enriched, bs)
        out = wrapper.predict_step(batch, 0)
        latents = _expect_tensordict(out["latents"])
        assert torch.equal(_expect_tensor(latents["mu"]), t2)
        assert torch.equal(_expect_tensor(latents["logvar"]), t3)

    def test_targets_are_cloned_not_view(self, bs: int, t1: Tensor, batch: TensorDict) -> None:
        """Targets in output are a clone of the original batch targets (Bug 1 fix).

        The clone must have the same values but be a different object, so that
        any in-place modifications during transform cannot corrupt the output targets.
        """
        enriched = TensorDict({"predictions": t1}, batch_size=[bs])
        wrapper = _make_wrapper(enriched, bs)
        out = wrapper.predict_step(batch, 0)
        # Values must match
        assert torch.equal(
            _expect_tensor(_expect_tensordict(out["targets"])["y"]),
            _expect_tensor(_expect_tensordict(batch["targets"])["y"]),
        )
        # Must be a different object (clone, not view)
        assert out["targets"] is not batch["targets"]

    def test_output_has_all_three_keys(self, bs: int, t1: Tensor, batch: TensorDict) -> None:
        """predict_step output always contains 'predictions', 'targets', and 'latents'."""
        enriched = TensorDict({"predictions": t1}, batch_size=[bs])
        wrapper = _make_wrapper(enriched, bs)
        out = wrapper.predict_step(batch, 0)
        assert "predictions" in out.keys()
        assert "targets" in out.keys()
        assert "latents" in out.keys()


# ---------------------------------------------------------------------------
# TestInverseTransformPredictions
# ---------------------------------------------------------------------------


class TestInverseTransformPredictions:
    def test_tensor_with_matching_chain_applies_inverse(self, t1: Tensor, bs: int) -> None:
        class _HalfTransform(torch.nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x * 2.0

            def inverse_transform(self, x: Tensor) -> Tensor:
                return x / 2.0

        transformer = NamedBatchTransformer(
            feature_chains={},
            target_chains={"y": _HalfTransform()},
        )
        result = transformer.inverse_transform_predictions(t1, "y")
        assert isinstance(result, Tensor)
        assert torch.allclose(result, t1 / 2.0)

    def test_tensor_with_no_matching_chain_unchanged(self, t1: Tensor) -> None:
        transformer = NamedBatchTransformer(feature_chains={}, target_chains={})
        result = transformer.inverse_transform_predictions(t1, "y")
        assert result is t1

    def test_tensordict_applies_per_key_inverse(self, bs: int, t1: Tensor, t2: Tensor) -> None:
        class _HalfTransform(torch.nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                return x * 2.0

            def inverse_transform(self, x: Tensor) -> Tensor:
                return x / 2.0

        transformer = NamedBatchTransformer(
            feature_chains={},
            target_chains={"a": _HalfTransform()},
        )
        preds = TensorDict({"a": t1, "b": t2}, batch_size=[bs])
        result = transformer.inverse_transform_predictions(preds, "ignored")
        assert isinstance(result, TensorDict)
        assert torch.allclose(_expect_tensor(result["a"]), t1 / 2.0)
        assert torch.equal(_expect_tensor(result["b"]), t2)

    def test_tensordict_no_matching_keys_passed_through(self, bs: int, t1: Tensor) -> None:
        transformer = NamedBatchTransformer(feature_chains={}, target_chains={})
        preds = TensorDict({"z": t1}, batch_size=[bs])
        result = transformer.inverse_transform_predictions(preds, "y")
        assert isinstance(result, TensorDict)
        assert torch.equal(_expect_tensor(result["z"]), t1)
