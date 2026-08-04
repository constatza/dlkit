"""Tests for model-family classification in model_detection.py."""

from __future__ import annotations

from types import SimpleNamespace

import lightning.pytorch as pl
from torch import nn

from dlkit.domain.nn.graph.base import BaseGraphNetwork
from dlkit.engine.adapters.lightning.model_detection import ModelType, detect_model_type


class _PlainModule(nn.Module):
    """A bare torch.nn.Module: neither Lightning-wrapped nor a dlkit graph network."""

    def forward(self, x):
        """Identity forward."""
        return x


class _ExternalLightningModule(pl.LightningModule):
    """A LightningModule that is not a dlkit graph network."""


class _GraphModule(BaseGraphNetwork):
    """A minimal BaseGraphNetwork subclass."""


def test_detect_model_type_graph_for_base_graph_network_subclass():
    settings = SimpleNamespace(name=_GraphModule)
    assert detect_model_type(settings) is ModelType.GRAPH


def test_detect_model_type_shape_agnostic_for_external_lightning_module():
    settings = SimpleNamespace(name=_ExternalLightningModule)
    assert detect_model_type(settings) is ModelType.SHAPE_AGNOSTIC_EXTERNAL


def test_detect_model_type_shape_aware_for_plain_nn_module():
    settings = SimpleNamespace(name=_PlainModule)
    assert detect_model_type(settings) is ModelType.SHAPE_AWARE_DLKIT


def test_detect_model_type_shape_agnostic_when_name_missing():
    settings = SimpleNamespace()
    assert detect_model_type(settings) is ModelType.SHAPE_AGNOSTIC_EXTERNAL


def test_detect_model_type_shape_agnostic_when_name_is_not_a_class():
    settings = SimpleNamespace(name={"some": "dict"})
    assert detect_model_type(settings) is ModelType.SHAPE_AGNOSTIC_EXTERNAL


def test_detect_model_type_shape_agnostic_on_unresolvable_import_path():
    settings = SimpleNamespace(name="NoSuchClass", module_path="dlkit.does.not.exist")
    assert detect_model_type(settings) is ModelType.SHAPE_AGNOSTIC_EXTERNAL


def test_detect_model_type_resolves_string_name_via_module_path():
    settings = SimpleNamespace(
        name="_GraphModule",
        module_path="tests.engine.adapters.lightning.test_model_detection",
    )
    assert detect_model_type(settings) is ModelType.GRAPH
