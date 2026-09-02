from __future__ import annotations

from pathlib import Path
from typing import get_args

from typer.testing import CliRunner

from dlkit.infrastructure.config._template_helpers import TemplateKind
from dlkit.interfaces.api import generate_template
from dlkit.interfaces.cli.commands.config import app as config_app


def _synced_paths(root: Path) -> list[Path]:
    return [root / "example_config.toml"] + [
        root / "config" / "templates" / f"{kind}.toml" for kind in get_args(TemplateKind)
    ]


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

    for path in _synced_paths(tmp_path):
        assert path.exists()

    assert (tmp_path / "example_config.toml").read_text() == generate_template("training")


def test_sync_templates_check_detects_drift(cli_runner: CliRunner, tmp_path: Path) -> None:
    # Create files with wrong content
    (tmp_path / "example_config.toml").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "example_config.toml").write_text('[run]\ntype = "train"\n')

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
    for kind in get_args(TemplateKind):
        (tmp_path / "config" / "templates" / f"{kind}.toml").write_text(generate_template(kind))

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
