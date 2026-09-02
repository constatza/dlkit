from __future__ import annotations

from typing import get_args

from dlkit.engine.workflows.entrypoints.templates import (
    generate_template as runtime_generate_template,
)
from dlkit.infrastructure.config._template_helpers import TemplateKind
from dlkit.interfaces.cli import templates as tmpl


def test_runtime_templates_delegate_to_central_builder() -> None:
    for kind in get_args(TemplateKind):
        assert runtime_generate_template(kind) == tmpl.render_template(kind)


def test_training_template_contains_expected_sections() -> None:
    content = tmpl.render_template("training")
    assert "[run]" in content
    assert "[model]" in content
    assert "[training]" in content
    assert "[training.trainer]" in content or "[training]" in content
    assert "[data]" in content


def test_fit_template_omits_training_section() -> None:
    content = tmpl.render_template("fit")
    assert "[run]" in content
    assert "[model]" in content
    assert "[data]" in content
    assert "[training]" not in content


def test_convergence_template_contains_convergence_section() -> None:
    content = tmpl.render_template("convergence")
    assert "[convergence]" in content
    assert "[training]" in content


def test_multirun_template_contains_multirun_section() -> None:
    content = tmpl.render_template("multirun")
    assert "[multirun]" in content
    assert "[[multirun.runs]]" in content
    assert "[model]" not in content


def test_model_introspection_produces_named_feature_entries() -> None:
    content = tmpl.render_template("training", model="FFNNDeepONet", module_path="dlkit.domain.nn")
    assert 'class = "FFNNDeepONet"' in content
    assert 'name = "branch"' in content
    assert 'name = "trunk"' in content
