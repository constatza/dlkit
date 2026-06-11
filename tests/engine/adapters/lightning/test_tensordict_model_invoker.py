"""Tests for TensorDictModelInvoker, ModelOutputSpec, and _build_invoker_from_entries.

Covers:
- ModelOutputSpec.all_out_keys() for single and multi-output models
- TensorDictModelInvoker positional dispatch: model args match in_keys order
- TensorDictModelInvoker kwarg dispatch: model receives named tensors
- TensorDictModelInvoker multi-output (VAE): named latent keys, no "0"/"1" hack
- _build_invoker_from_entries: named features use kwarg dispatch (name == forward arg)
- _build_invoker_from_entries: kwarg dispatch ignores config-list order
- _build_invoker_from_entries: model_input=True (include), model_input=False (exclude)
- _build_invoker_from_entries: build-time signature validation via InvokerBuildResult.validator
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from tensordict import TensorDict
from torch import Tensor, nn

from dlkit.engine.adapters.lightning.model_invoker import (
    InvokerBuildResult,
    ModelOutputSpec,
    TensorDictModelInvoker,
    _build_invoker_from_entries,
)
from dlkit.infrastructure.config.data_roles import DataRole
from dlkit.infrastructure.config.entry_base import DataEntry
from dlkit.infrastructure.config.entry_types import ValueEntry

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
    """Named features dispatch as kwargs; entry name == forward() parameter name."""

    @pytest.fixture
    def feat_x(self, bs: int) -> DataEntry:
        """Feature 'x' with default model_input=True."""
        return ValueEntry(name="x", value=torch.zeros(bs, 3), data_role=DataRole.FEATURE)

    @pytest.fixture
    def feat_z(self, bs: int) -> DataEntry:
        """Feature 'z' with default model_input=True."""
        return ValueEntry(name="z", value=torch.zeros(bs, 5), data_role=DataRole.FEATURE)

    def test_returns_invoker_build_result(self, feat_x: DataEntry, feat_z: DataEntry) -> None:
        """_build_invoker_from_entries returns an InvokerBuildResult."""
        result = _build_invoker_from_entries([feat_x, feat_z])
        assert isinstance(result, InvokerBuildResult)
        assert isinstance(result.invoker, TensorDictModelInvoker)
        assert callable(result.validator)
        assert isinstance(result.forward_arg_map, dict)

    def test_forward_arg_map_contains_entry_names(
        self, feat_x: DataEntry, feat_z: DataEntry
    ) -> None:
        """forward_arg_map is identity mapping {name: name} for named features."""
        result = _build_invoker_from_entries([feat_x, feat_z])
        assert result.forward_arg_map == {"x": "x", "z": "z"}

    def test_named_features_dispatch_as_kwargs(
        self,
        bs: int,
        two_feature_batch: TensorDict,
        x_tensor: Tensor,
        z_tensor: Tensor,
        feat_x: DataEntry,
        feat_z: DataEntry,
    ) -> None:
        """Named model-input features are dispatched as kwargs; name == forward arg."""
        received: list[dict[str, Tensor]] = []

        class _KwargModel(nn.Module):
            def forward(self, x: Tensor, z: Tensor) -> Tensor:
                received.append({"x": x, "z": z})
                return torch.zeros(bs, 1)

        result = _build_invoker_from_entries([feat_x, feat_z])
        result.validator(_KwargModel())
        result.invoker.invoke(_KwargModel(), two_feature_batch)
        assert torch.equal(received[0]["x"], x_tensor)
        assert torch.equal(received[0]["z"], z_tensor)

    def test_kwarg_dispatch_ignores_config_order(
        self,
        bs: int,
        two_feature_batch: TensorDict,
        x_tensor: Tensor,
        z_tensor: Tensor,
        feat_x: DataEntry,
        feat_z: DataEntry,
    ) -> None:
        """Kwarg dispatch binds by name — config-list order does not affect routing."""
        received: list[dict[str, Tensor]] = []

        class _Model(nn.Module):
            def forward(self, x: Tensor, z: Tensor) -> Tensor:
                received.append({"x": x, "z": z})
                return torch.zeros(bs, 1)

        # z declared first in the list, x second — does not matter
        result = _build_invoker_from_entries([feat_z, feat_x])
        result.validator(_Model())
        result.invoker.invoke(_Model(), two_feature_batch)
        assert torch.equal(received[0]["x"], x_tensor)
        assert torch.equal(received[0]["z"], z_tensor)

    def test_kwarg_only_forward_works_with_named_dispatch(
        self,
        bs: int,
        two_feature_batch: TensorDict,
        feat_x: DataEntry,
        feat_z: DataEntry,
    ) -> None:
        """Keyword-only forward() parameters are valid with named dispatch."""

        class _KwOnlyModel(nn.Module):
            def forward(self, *, x: Tensor, z: Tensor) -> Tensor:
                return torch.zeros(bs, 1)

        result = _build_invoker_from_entries([feat_x, feat_z])
        result.validator(_KwOnlyModel())  # must not raise
        result.invoker.invoke(_KwOnlyModel(), two_feature_batch)

    def test_model_input_false_excludes_feature(
        self,
        bs: int,
        two_feature_batch: TensorDict,
        x_tensor: Tensor,
    ) -> None:
        """model_input=False excludes feature from model call."""
        feat_x = ValueEntry(
            name="x", value=torch.zeros(bs, 3), model_input=True, data_role=DataRole.FEATURE
        )
        feat_z = ValueEntry(
            name="z", value=torch.zeros(bs, 5), model_input=False, data_role=DataRole.FEATURE
        )
        received: list[Tensor] = []

        class _Rec(nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                received.append(x)
                return torch.zeros(bs, 1)

        result = _build_invoker_from_entries([feat_x, feat_z])
        result.validator(_Rec())
        result.invoker.invoke(_Rec(), two_feature_batch)
        assert len(received) == 1
        assert torch.equal(received[0], x_tensor)

    def test_no_model_input_raises_value_error(self, bs: int) -> None:
        """All model_input=False raises ValueError (no inputs to pass)."""
        feat_x = ValueEntry(
            name="x", value=torch.zeros(bs, 3), model_input=False, data_role=DataRole.FEATURE
        )
        with pytest.raises(ValueError, match="No model-input features"):
            _build_invoker_from_entries([feat_x])

    def test_output_spec_threaded_through(self, bs: int, simple_batch: TensorDict) -> None:
        """output_spec is used to set out_keys on the invoker."""
        feat_x = ValueEntry(
            name="x", value=torch.zeros(bs, 3), model_input=True, data_role=DataRole.FEATURE
        )
        spec = ModelOutputSpec(latent_keys=(("latents", "z"),))

        class _Rec(nn.Module):
            def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
                return torch.zeros(bs, 2), torch.zeros(bs, 3)

        result = _build_invoker_from_entries([feat_x], output_spec=spec)
        result.validator(_Rec())
        out = result.invoker.invoke(_Rec(), simple_batch)
        assert "latents" in out.keys()
        assert "z" in cast(TensorDict, out["latents"]).keys()


class TestForwardSignatureValidation:
    """Validation raised at build time for unsafe or mismatched signatures."""

    @pytest.fixture
    def bs(self) -> int:
        return 4

    @pytest.fixture
    def feat_x(self, bs: int) -> DataEntry:
        return ValueEntry(name="x", value=torch.zeros(bs, 3), data_role=DataRole.FEATURE)

    @pytest.fixture
    def feat_z(self, bs: int) -> DataEntry:
        return ValueEntry(name="z", value=torch.zeros(bs, 5), data_role=DataRole.FEATURE)

    def test_mismatched_name_raises_value_error(self, feat_x: DataEntry, feat_z: DataEntry) -> None:
        """Feature name not in forward() signature raises ValueError at validation time."""

        class _Model(nn.Module):
            def forward(self, a: Tensor, b: Tensor) -> Tensor:
                return torch.zeros(1)

        result = _build_invoker_from_entries([feat_x, feat_z])
        with pytest.raises(ValueError, match="x"):
            result.validator(_Model())

    def test_var_positional_raises_type_error(self, feat_x: DataEntry) -> None:
        """*args in forward() raises TypeError at validation time."""

        class _Model(nn.Module):
            def forward(self, *args: Tensor) -> Tensor:
                return torch.zeros(1)

        result = _build_invoker_from_entries([feat_x])
        with pytest.raises(TypeError, match=r"\*args"):
            result.validator(_Model())

    def test_var_keyword_raises_type_error(self, feat_x: DataEntry) -> None:
        """**kwargs in forward() raises TypeError at validation time."""

        class _Model(nn.Module):
            def forward(self, **kwargs: Tensor) -> Tensor:
                return torch.zeros(1)

        result = _build_invoker_from_entries([feat_x])
        with pytest.raises(TypeError, match=r"\*\*kwargs"):
            result.validator(_Model())
