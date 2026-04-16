"""Tests for TensorDictModelInvoker, ModelOutputSpec, and _build_invoker_from_entries.

Covers:
- ModelOutputSpec.all_out_keys() for single and multi-output models
- TensorDictModelInvoker positional dispatch: model args match in_keys order
- TensorDictModelInvoker kwarg dispatch: model receives named tensors
- TensorDictModelInvoker multi-output (VAE): named latent keys, no "0"/"1" hack
- _build_invoker_from_entries: model_input=True (kwarg), int, str, False/None
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from tensordict import TensorDict
from torch import Tensor, nn

from dlkit.engine.adapters.lightning.model_invoker import (
    ModelOutputSpec,
    TensorDictModelInvoker,
    _build_invoker_from_entries,
)
from dlkit.infrastructure.config.data_entries import Feature, FeatureType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bs() -> int:
    """Standard batch size."""
    return 4


@pytest.fixture
def x_tensor(bs: int) -> Tensor:
    """Feature tensor for entry 'x': (bs, 3)."""
    return torch.randn(bs, 3)


@pytest.fixture
def z_tensor(bs: int) -> Tensor:
    """Feature tensor for entry 'z': (bs, 5)."""
    return torch.randn(bs, 5)


@pytest.fixture
def simple_batch(bs: int, x_tensor: Tensor) -> TensorDict:
    """Minimal batch with a single 'x' feature."""
    return TensorDict(
        {"features": TensorDict({"x": x_tensor}, batch_size=[bs])},
        batch_size=[bs],
    )


@pytest.fixture
def two_feature_batch(bs: int, x_tensor: Tensor, z_tensor: Tensor) -> TensorDict:
    """Batch with two features 'x' and 'z'."""
    return TensorDict(
        {
            "features": TensorDict(
                {"x": x_tensor, "z": z_tensor},
                batch_size=[bs],
            )
        },
        batch_size=[bs],
    )


# ---------------------------------------------------------------------------
# TestModelOutputSpec
# ---------------------------------------------------------------------------


class TestModelOutputSpec:
    def test_default_spec_has_only_predictions(self) -> None:
        spec = ModelOutputSpec()
        assert spec.prediction_key == "predictions"
        assert spec.latent_keys == ()

    def test_all_out_keys_single_output(self) -> None:
        spec = ModelOutputSpec()
        assert spec.all_out_keys() == ["predictions"]

    def test_all_out_keys_with_flat_latent(self) -> None:
        spec = ModelOutputSpec(latent_keys=("latents",))
        assert spec.all_out_keys() == ["predictions", "latents"]

    def test_all_out_keys_with_nested_latents(self) -> None:
        spec = ModelOutputSpec(latent_keys=(("latents", "mu"), ("latents", "logvar")))
        assert spec.all_out_keys() == [
            "predictions",
            ("latents", "mu"),
            ("latents", "logvar"),
        ]

    def test_custom_prediction_key(self) -> None:
        spec = ModelOutputSpec(prediction_key="recon")
        assert spec.all_out_keys() == ["recon"]

    def test_spec_is_frozen(self) -> None:
        spec = ModelOutputSpec()
        with pytest.raises(Exception):
            cast(Any, spec).prediction_key = "other"


# ---------------------------------------------------------------------------
# TestTensorDictModelInvoker — single-feature, single-output
# ---------------------------------------------------------------------------


class TestTensorDictModelInvokerSingleFeature:
    def test_invoke_returns_tensordict(
        self, bs: int, simple_batch: TensorDict, x_tensor: Tensor
    ) -> None:
        """invoke() must return a TensorDict (enriched batch)."""
        model = nn.Linear(3, 2)
        invoker = TensorDictModelInvoker(in_keys=[("features", "x")])
        result = invoker.invoke(model, simple_batch)
        assert isinstance(result, TensorDict)

    def test_predictions_key_present(self, bs: int, simple_batch: TensorDict) -> None:
        """Enriched batch has 'predictions' key after invoke()."""
        model = nn.Linear(3, 2)
        invoker = TensorDictModelInvoker(in_keys=[("features", "x")])
        result = invoker.invoke(model, simple_batch)
        assert "predictions" in result.keys()

    def test_predictions_shape(self, bs: int, simple_batch: TensorDict) -> None:
        """Prediction tensor has expected output shape."""
        model = nn.Linear(3, 2)
        invoker = TensorDictModelInvoker(in_keys=[("features", "x")])
        result = invoker.invoke(model, simple_batch)
        assert result["predictions"].shape == (bs, 2)

    def test_original_keys_preserved(self, bs: int, simple_batch: TensorDict) -> None:
        """Features and other existing keys are still in the returned batch."""
        model = nn.Linear(3, 2)
        invoker = TensorDictModelInvoker(in_keys=[("features", "x")])
        result = invoker.invoke(model, simple_batch)
        assert "features" in result.keys()
        assert "x" in cast(TensorDict, result["features"]).keys()


# ---------------------------------------------------------------------------
# TestTensorDictModelInvoker — positional ordering
# ---------------------------------------------------------------------------


class TestTensorDictModelInvokerOrdering:
    def test_positional_order_matches_in_keys(
        self, bs: int, two_feature_batch: TensorDict, x_tensor: Tensor, z_tensor: Tensor
    ) -> None:
        """Model receives features in the order declared by in_keys."""
        received: list[tuple[Tensor, Tensor]] = []

        class _RecordingModel(nn.Module):
            def forward(self, x: Tensor, z: Tensor) -> Tensor:
                received.append((x, z))
                return torch.zeros(bs, 1)

        invoker = TensorDictModelInvoker(in_keys=[("features", "x"), ("features", "z")])
        invoker.invoke(_RecordingModel(), two_feature_batch)
        assert len(received) == 1
        assert torch.equal(received[0][0], x_tensor)
        assert torch.equal(received[0][1], z_tensor)

    def test_reversed_order_swaps_args(
        self, bs: int, two_feature_batch: TensorDict, x_tensor: Tensor, z_tensor: Tensor
    ) -> None:
        """Reversing in_keys reverses positional args to the model."""
        received: list[tuple[Tensor, Tensor]] = []

        class _RecordingModel(nn.Module):
            def forward(self, first: Tensor, second: Tensor) -> Tensor:
                received.append((first, second))
                return torch.zeros(bs, 1)

        invoker = TensorDictModelInvoker(in_keys=[("features", "z"), ("features", "x")])
        invoker.invoke(_RecordingModel(), two_feature_batch)
        assert torch.equal(received[0][0], z_tensor)  # z is first
        assert torch.equal(received[0][1], x_tensor)  # x is second


# ---------------------------------------------------------------------------
# TestTensorDictModelInvoker — multi-output (VAE-style)
# ---------------------------------------------------------------------------


class TestTensorDictModelInvokerMultiOutput:
    @pytest.fixture
    def vae_output_spec(self) -> ModelOutputSpec:
        """Output spec for a VAE: predictions + mu + logvar."""
        return ModelOutputSpec(latent_keys=(("latents", "mu"), ("latents", "logvar")))

    def test_named_latents_in_enriched_batch(
        self,
        bs: int,
        simple_batch: TensorDict,
        vae_output_spec: ModelOutputSpec,
    ) -> None:
        """Named latent keys are written correctly — no '0'/'1' positional hack."""

        class _VAEModel(nn.Module):
            def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
                return (
                    torch.zeros(bs, 2),  # recon
                    torch.zeros(bs, 4),  # mu
                    torch.zeros(bs, 4),  # logvar
                )

        invoker = TensorDictModelInvoker(
            in_keys=[("features", "x")],
            output_spec=vae_output_spec,
        )
        result = invoker.invoke(_VAEModel(), simple_batch)

        assert "predictions" in result.keys()
        assert result["predictions"].shape == (bs, 2)
        # Named latents — NOT "0"/"1"
        assert "latents" in result.keys()
        latents = cast(TensorDict, result["latents"])
        assert "mu" in latents.keys()
        assert "logvar" in latents.keys()
        assert "0" not in result.keys()
        assert "1" not in result.keys()

    def test_latents_shapes(
        self,
        bs: int,
        simple_batch: TensorDict,
        vae_output_spec: ModelOutputSpec,
    ) -> None:
        """Named latent tensors have the correct shapes."""

        class _VAEModel(nn.Module):
            def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
                return torch.zeros(bs, 2), torch.zeros(bs, 4), torch.zeros(bs, 4)

        invoker = TensorDictModelInvoker(
            in_keys=[("features", "x")],
            output_spec=vae_output_spec,
        )
        result = invoker.invoke(_VAEModel(), simple_batch)
        latents = cast(TensorDict, result["latents"])
        assert latents["mu"].shape == (bs, 4)
        assert latents["logvar"].shape == (bs, 4)


# ---------------------------------------------------------------------------
# TestBuildInvokerFromEntries
# ---------------------------------------------------------------------------


class TestBuildInvokerFromEntries:
    @pytest.fixture
    def feat_x(self) -> FeatureType:
        """Feature 'x' with default model_input=True."""
        return Feature("x", value=torch.zeros(4, 3))

    @pytest.fixture
    def feat_z(self) -> FeatureType:
        """Feature 'z' with default model_input=True."""
        return Feature("z", value=torch.zeros(4, 5))

    def test_model_input_true_dispatches_by_entry_name(
        self,
        bs: int,
        two_feature_batch: TensorDict,
        x_tensor: Tensor,
        z_tensor: Tensor,
    ) -> None:
        """model_input=True (default): features passed as kwargs using entry name as key."""
        feat_x = Feature("x", value=torch.zeros(bs, 3), model_input=True)
        feat_z = Feature("z", value=torch.zeros(bs, 5), model_input=True)
        received: dict[str, Tensor] = {}

        class _KwModel(nn.Module):
            def forward(self, x: Tensor, z: Tensor) -> Tensor:
                received.update({"x": x, "z": z})
                return torch.zeros(bs, 1)

        invoker = _build_invoker_from_entries([feat_x, feat_z])
        invoker.invoke(_KwModel(), two_feature_batch)
        assert torch.equal(received["x"], x_tensor)
        assert torch.equal(received["z"], z_tensor)

    def test_explicit_int_ordering_overrides_insertion_order(
        self,
        bs: int,
        two_feature_batch: TensorDict,
        x_tensor: Tensor,
        z_tensor: Tensor,
    ) -> None:
        """model_input=int: features sorted by declared index, not insertion order."""
        # Declared x as index 1, z as index 0 → model(z, x) even though x is first
        feat_x = Feature("x", value=torch.zeros(bs, 3), model_input=1)
        feat_z = Feature("z", value=torch.zeros(bs, 5), model_input=0)

        received: list[tuple[Tensor, Tensor]] = []

        class _Rec(nn.Module):
            def forward(self, first: Tensor, second: Tensor) -> Tensor:
                received.append((first, second))
                return torch.zeros(bs, 1)

        # Order in list is [x, z] but model_input says z=0, x=1
        invoker = _build_invoker_from_entries([feat_x, feat_z])
        invoker.invoke(_Rec(), two_feature_batch)
        assert torch.equal(received[0][0], z_tensor)  # index 0 → first
        assert torch.equal(received[0][1], x_tensor)  # index 1 → second

    def test_explicit_digit_str_ordering_overrides_insertion_order(
        self,
        bs: int,
        two_feature_batch: TensorDict,
        x_tensor: Tensor,
        z_tensor: Tensor,
    ) -> None:
        """model_input='0'/'1' (digit strings): same as int positional, useful in TOML."""
        feat_x = Feature("x", value=torch.zeros(bs, 3), model_input="1")
        feat_z = Feature("z", value=torch.zeros(bs, 5), model_input="0")

        received: list[tuple[Tensor, Tensor]] = []

        class _Rec(nn.Module):
            def forward(self, first: Tensor, second: Tensor) -> Tensor:
                received.append((first, second))
                return torch.zeros(bs, 1)

        invoker = _build_invoker_from_entries([feat_x, feat_z])
        invoker.invoke(_Rec(), two_feature_batch)
        assert torch.equal(received[0][0], z_tensor)  # index 0 → first
        assert torch.equal(received[0][1], x_tensor)  # index 1 → second

    def test_model_input_str_kwarg_dispatch(
        self,
        bs: int,
        two_feature_batch: TensorDict,
        x_tensor: Tensor,
        z_tensor: Tensor,
    ) -> None:
        """model_input='name': feature passed as kwarg with custom name."""
        feat_x = Feature("x", value=torch.zeros(bs, 3), model_input="x_input")
        feat_z = Feature("z", value=torch.zeros(bs, 5), model_input="z_input")
        received: dict[str, Tensor] = {}

        class _KwModel(nn.Module):
            def forward(self, x_input: Tensor, z_input: Tensor) -> Tensor:
                received.update({"x_input": x_input, "z_input": z_input})
                return torch.zeros(bs, 1)

        invoker = _build_invoker_from_entries([feat_x, feat_z])
        invoker.invoke(_KwModel(), two_feature_batch)
        assert torch.equal(received["x_input"], x_tensor)
        assert torch.equal(received["z_input"], z_tensor)

    def test_mixed_positional_and_kwarg(
        self,
        bs: int,
        two_feature_batch: TensorDict,
        x_tensor: Tensor,
        z_tensor: Tensor,
    ) -> None:
        """model_input=0 positional + model_input='extra' kwarg — mixed dispatch."""
        feat_x = Feature("x", value=torch.zeros(bs, 3), model_input=0)  # positional
        feat_z = Feature("z", value=torch.zeros(bs, 5), model_input="extra")  # kwarg
        received: dict = {}

        class _Mixed(nn.Module):
            def forward(self, pos: Tensor, extra: Tensor) -> Tensor:
                received.update({"pos": pos, "extra": extra})
                return torch.zeros(bs, 1)

        invoker = _build_invoker_from_entries([feat_x, feat_z])
        invoker.invoke(_Mixed(), two_feature_batch)
        assert torch.equal(received["pos"], x_tensor)
        assert torch.equal(received["extra"], z_tensor)

    def test_model_input_false_excludes_feature(
        self,
        bs: int,
        two_feature_batch: TensorDict,
        x_tensor: Tensor,
    ) -> None:
        """model_input=False excludes feature from model call."""
        feat_x = Feature("x", value=torch.zeros(bs, 3), model_input=True)
        feat_z = Feature("z", value=torch.zeros(bs, 5), model_input=False)

        received: list[Tensor] = []

        class _Rec(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                received.append(x)
                return torch.zeros(bs, 1)

        invoker = _build_invoker_from_entries([feat_x, feat_z])
        invoker.invoke(_Rec(), two_feature_batch)
        assert len(received) == 1
        assert torch.equal(received[0], x_tensor)

    def test_model_input_none_excludes_feature(
        self,
        bs: int,
        two_feature_batch: TensorDict,
        x_tensor: Tensor,
    ) -> None:
        """model_input=None excludes feature from model call."""
        feat_x = Feature("x", value=torch.zeros(bs, 3), model_input=True)
        feat_z = Feature("z", value=torch.zeros(bs, 5), model_input=None)

        received: list[Tensor] = []

        class _Rec(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                received.append(x)
                return torch.zeros(bs, 1)

        invoker = _build_invoker_from_entries([feat_x, feat_z])
        invoker.invoke(_Rec(), two_feature_batch)
        assert len(received) == 1

    def test_no_model_input_raises_value_error(self) -> None:
        """All model_input=False raises ValueError (no inputs to pass)."""
        feat_x = Feature("x", value=torch.zeros(4, 3), model_input=False)
        with pytest.raises(ValueError, match="No model-input features"):
            _build_invoker_from_entries([feat_x])

    def test_output_spec_threaded_through(self, bs: int, simple_batch: TensorDict) -> None:
        """output_spec is used to set out_keys on the invoker."""
        feat_x = Feature("x", value=torch.zeros(bs, 3), model_input=True)
        spec = ModelOutputSpec(latent_keys=(("latents", "z"),))

        class _Rec(nn.Module):
            def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
                return torch.zeros(bs, 2), torch.zeros(bs, 3)

        invoker = _build_invoker_from_entries([feat_x], output_spec=spec)
        result = invoker.invoke(_Rec(), simple_batch)
        assert "latents" in result.keys()
        assert "z" in cast(TensorDict, result["latents"]).keys()
