from __future__ import annotations

from dlkit.engine.workflows.entrypoints.templates import (
    generate_template as runtime_generate_template,
)
from dlkit.interfaces.cli import templates as tmpl


def test_runtime_templates_delegate_to_central_builder() -> None:
    assert runtime_generate_template("training") == tmpl.render_template("training")
    assert runtime_generate_template("inference") == tmpl.render_template("inference")
    assert runtime_generate_template("mlflow") == tmpl.render_template("mlflow")
    assert runtime_generate_template("optuna") == tmpl.render_template("optuna")


def test_training_template_contains_expected_sections() -> None:
    content = tmpl.render_template("training")
    assert "[run]" in content
    assert "[model]" in content
    assert "[training]" in content
    assert "[training.trainer]" in content or "[training]" in content
    assert "[data]" in content
