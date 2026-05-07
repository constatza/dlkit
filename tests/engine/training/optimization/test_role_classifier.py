"""Tests for graph-based parameter role classification."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from dlkit.domain.nn.parameter_roles import ParameterRole
from dlkit.engine.training.optimization.role_classifier import (
    GraphParameterRoleClassifier,
    classify_parameter_roles,
)


class _ThreeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc0 = nn.Linear(4, 8)
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc0(x))
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class _Embedder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc0 = nn.Linear(4, 8, bias=False)
        self.fc1 = nn.Linear(8, 8, bias=False)
        self.fc2 = nn.Linear(8, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc0(x))
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class _CompositeWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedder = _Embedder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedder(x)


class _ResidualModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_proj = nn.Linear(4, 8, bias=False)
        self.res0 = nn.Linear(8, 8, bias=False)
        self.res1 = nn.Linear(8, 8, bias=False)
        self.output_proj = nn.Linear(8, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.input_proj(x))
        residual = x
        x = torch.relu(self.res0(x))
        x = self.res1(x) + residual
        return self.output_proj(torch.relu(x))


class _MultiHeadModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_proj = nn.Linear(4, 8, bias=False)
        self.trunk = nn.Linear(8, 8, bias=False)
        self.head0 = nn.Linear(8, 2, bias=False)
        self.head1 = nn.Linear(8, 2, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.input_proj(x))
        trunk = torch.relu(self.trunk(x))
        return self.head0(trunk), self.head1(trunk)


class _EmbeddingHeadModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.hidden = nn.Linear(8, 8, bias=False)
        self.head = nn.Linear(8, 16, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        x = torch.relu(self.hidden(x))
        return self.head(x)


class _FunctionalLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(8, 4))
        self.b0 = nn.Parameter(torch.randn(8))
        self.w1 = nn.Parameter(torch.randn(8, 8))
        self.w2 = nn.Parameter(torch.randn(2, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(F.linear(x, self.w0, self.b0))
        x = torch.relu(F.linear(x, self.w1))
        return F.linear(x, self.w2)


class _FunctionalConvModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(4, 2, 1))
        self.b0 = nn.Parameter(torch.randn(4))
        self.w1 = nn.Parameter(torch.randn(3, 4, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(F.conv1d(x, self.w0, self.b0))
        return F.conv1d(x, self.w1)


class _UnsupportedFunctionalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.t())


class _SharedWeightModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(4, 4, bias=False)
        self.second = nn.Linear(4, 4, bias=False)
        self.second.weight = self.first.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.first(x))
        return self.second(x)


class _TraceFailureModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:
            return self.fc(x)
        return -self.fc(x)


class _ModuleListModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(4, 8, bias=False),
                nn.Linear(8, 8, bias=False),
                nn.Linear(8, 2, bias=False),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < 2:
                x = torch.relu(x)
        return x


@pytest.fixture
def classifier() -> GraphParameterRoleClassifier:
    return GraphParameterRoleClassifier()


def test_three_layer_model_uses_input_hidden_output_roles(
    classifier: GraphParameterRoleClassifier,
) -> None:
    roles = classifier.classify(_ThreeLayer())

    assert roles["fc0.weight"] == ParameterRole.INPUT
    assert roles["fc1.weight"] == ParameterRole.HIDDEN
    assert roles["fc2.weight"] == ParameterRole.OUTPUT
    assert roles["fc0.bias"] == ParameterRole.BIAS
    assert roles["fc1.bias"] == ParameterRole.BIAS
    assert roles["fc2.bias"] == ParameterRole.BIAS


def test_composite_wrapper_classifies_fundamental_sublayers_only(
    classifier: GraphParameterRoleClassifier,
) -> None:
    roles = classifier.classify(_CompositeWrapper())

    assert roles["embedder.fc0.weight"] == ParameterRole.INPUT
    assert roles["embedder.fc1.weight"] == ParameterRole.HIDDEN
    assert roles["embedder.fc2.weight"] == ParameterRole.OUTPUT


def test_residual_model_keeps_interior_sites_hidden(
    classifier: GraphParameterRoleClassifier,
) -> None:
    roles = classifier.classify(_ResidualModel())

    assert roles["input_proj.weight"] == ParameterRole.INPUT
    assert roles["res0.weight"] == ParameterRole.HIDDEN
    assert roles["res1.weight"] == ParameterRole.HIDDEN
    assert roles["output_proj.weight"] == ParameterRole.OUTPUT


def test_multi_head_model_marks_terminal_heads_as_output(
    classifier: GraphParameterRoleClassifier,
) -> None:
    roles = classifier.classify(_MultiHeadModel())

    assert roles["input_proj.weight"] == ParameterRole.INPUT
    assert roles["trunk.weight"] == ParameterRole.HIDDEN
    assert roles["head0.weight"] == ParameterRole.OUTPUT
    assert roles["head1.weight"] == ParameterRole.OUTPUT


def test_module_list_classification_follows_execution_graph(
    classifier: GraphParameterRoleClassifier,
) -> None:
    roles = classify_parameter_roles(_ModuleListModel())

    assert roles["layers.0.weight"] == ParameterRole.INPUT
    assert roles["layers.1.weight"] == ParameterRole.HIDDEN
    assert roles["layers.2.weight"] == ParameterRole.OUTPUT


def test_embedding_and_head_receive_structural_and_output_roles(
    classifier: GraphParameterRoleClassifier,
) -> None:
    roles = classifier.classify(_EmbeddingHeadModel())

    assert roles["embed.weight"] == ParameterRole.EMBEDDING
    assert roles["hidden.weight"] == ParameterRole.HIDDEN
    assert roles["head.weight"] == ParameterRole.OUTPUT


def test_supported_functional_linear_sites_are_classified(
    classifier: GraphParameterRoleClassifier,
) -> None:
    roles = classifier.classify(_FunctionalLinearModel())

    assert roles["w0"] == ParameterRole.INPUT
    assert roles["b0"] == ParameterRole.BIAS
    assert roles["w1"] == ParameterRole.HIDDEN
    assert roles["w2"] == ParameterRole.OUTPUT


def test_supported_functional_conv_sites_are_classified(
    classifier: GraphParameterRoleClassifier,
) -> None:
    roles = classifier.classify(_FunctionalConvModel())

    assert roles["w0"] == ParameterRole.INPUT
    assert roles["b0"] == ParameterRole.BIAS
    assert roles["w1"] == ParameterRole.OUTPUT


def test_unsupported_functional_parameter_pattern_falls_back_to_unknown(
    classifier: GraphParameterRoleClassifier,
) -> None:
    roles = classifier.classify(_UnsupportedFunctionalModel())
    assert roles["weight"] == ParameterRole.UNKNOWN


def test_shared_parameter_is_classified_as_unknown(
    classifier: GraphParameterRoleClassifier,
) -> None:
    roles = classifier.classify(_SharedWeightModel())
    assert roles["first.weight"] == ParameterRole.UNKNOWN


def test_trace_failure_keeps_structural_roles_and_marks_remaining_unknown(
    classifier: GraphParameterRoleClassifier,
) -> None:
    roles = classifier.classify(_TraceFailureModel())
    assert roles["fc.weight"] == ParameterRole.UNKNOWN
    assert roles["fc.bias"] == ParameterRole.BIAS
