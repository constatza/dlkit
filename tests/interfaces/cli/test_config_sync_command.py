from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dlkit.interfaces.api import generate_template
from dlkit.interfaces.cli.commands.config import app as config_app


def test_sync_templates_write_creates_files(cli_runner: CliRunner, tmp_path: Path) -> None:
    result = cli_runner.invoke(
        config_app,
        [
            "sync-templates",
            "--root",
            str(tmp_path),
            "--write",
        ],
    )
    assert result.exit_code == 0

    for name in [
        tmp_path / "example_config.toml",
        tmp_path / "config" / "templates" / "training.toml",
        tmp_path / "config" / "templates" / "inference.toml",
        tmp_path / "config" / "templates" / "mlflow.toml",
        tmp_path / "config" / "templates" / "optuna.toml",
    ]:
        assert name.exists()

    assert (tmp_path / "example_config.toml").read_text() == generate_template("training")


def test_sync_templates_check_detects_drift(cli_runner: CliRunner, tmp_path: Path) -> None:
    # Create files with wrong content
    (tmp_path / "example_config.toml").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "example_config.toml").write_text("[SESSION]\nname = 'wrong'\n")

    result = cli_runner.invoke(
        config_app,
        [
            "sync-templates",
            "--root",
            str(tmp_path),
            "--check",
        ],
    )
    assert result.exit_code == 1


def test_sync_templates_check_ok(cli_runner: CliRunner, tmp_path: Path) -> None:
    # Seed with correct files
    (tmp_path / "config" / "templates").mkdir(parents=True, exist_ok=True)
    (tmp_path / "example_config.toml").write_text(generate_template("training"))
    (tmp_path / "config" / "templates" / "training.toml").write_text(generate_template("training"))
    (tmp_path / "config" / "templates" / "inference.toml").write_text(
        generate_template("inference")
    )
    (tmp_path / "config" / "templates" / "mlflow.toml").write_text(generate_template("mlflow"))
    (tmp_path / "config" / "templates" / "optuna.toml").write_text(generate_template("optuna"))

    result = cli_runner.invoke(
        config_app,
        [
            "sync-templates",
            "--root",
            str(tmp_path),
            "--check",
        ],
    )
    assert result.exit_code == 0
