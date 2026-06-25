from __future__ import annotations

from dlkit.interfaces.cli import templates as tmpl
from dlkit.interfaces.cli.commands.config import (
    _create_inference_template,
    _create_mlflow_template,
    _create_optuna_template,
    _create_training_template,
)


def test_cli_templates_use_central_builder() -> None:
    assert _create_training_template() == tmpl.render_template("training")
    assert _create_inference_template() == tmpl.render_template("inference")
    assert _create_mlflow_template() == tmpl.render_template("mlflow")
    assert _create_optuna_template() == tmpl.render_template("optuna")


def test_training_template_contains_expected_sections() -> None:
    content = tmpl.render_template("training")
    assert "[run]" in content
    assert "[model]" in content
    assert "[training]" in content
    assert "[training.trainer]" in content or "[training]" in content
    assert "[data]" in content
