"""Tests for the `dlkit multirun` CLI command group (run/validate)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from dlkit.common.results import ChildFailure, ChildSuccess, MultiRunResult, TrainingResult
from dlkit.engine.workflows.multi_run import ExplicitFileSource, RunSpec
from dlkit.interfaces.cli.app import app as cli_app


@pytest.fixture
def multirun_config_path(tmp_path: Path) -> Path:
    """A minimal `[run] type = "multirun"` TOML file — never actually parsed.

    `load_config` is patched in every test below, so this file's contents
    don't matter beyond existing and being readable (`CONFIG_PATH_ARG`
    requires that).
    """
    config_path = tmp_path / "sweep.toml"
    config_path.write_text('[run]\ntype = "multirun"\n')
    return config_path


@pytest.fixture
def mock_multirun_job() -> SimpleNamespace:
    """Minimal MultiRunJobConfig-shaped stand-in for CLI-level tests."""
    return SimpleNamespace(
        parent_run_name="sweep-parent",
        experiment_name="sweep-experiment",
        failure_policy="fail_fast",
        runs=[SimpleNamespace(id="a"), SimpleNamespace(id="b")],
    )


@pytest.fixture
def mock_multirun_result() -> MultiRunResult:
    """A real MultiRunResult with one success and one failure child.

    Built from real dataclasses (not Mocks) since the CLI presenter
    pattern-matches on the concrete ChildSuccess/ChildFailure types.
    """
    success = ChildSuccess(
        child_id="a",
        label="a",
        run_id="run-a",
        result=TrainingResult(model_state=None, metrics={}, artifacts={}, duration_seconds=0.1),
    )
    failure = ChildFailure(
        child_id="b",
        label="b",
        exception_type="ValueError",
        message="boom",
        run_id=None,
        stage="execute",
    )
    return MultiRunResult(parent_run_id="parent-run-id", children=(success, failure))


class TestMultirunRunCommand:
    """`dlkit multirun run CONFIG.toml`."""

    def test_run_invokes_api_and_presents_result(
        self,
        cli_runner: CliRunner,
        multirun_config_path: Path,
        mock_multirun_job: SimpleNamespace,
        mock_multirun_result: MultiRunResult,
    ) -> None:
        with (
            patch(
                "dlkit.interfaces.cli.commands.multirun.load_config",
                return_value=mock_multirun_job,
            ) as mock_load,
            patch(
                "dlkit.interfaces.cli.commands.multirun.api_run_multirun_config",
                return_value=mock_multirun_result,
            ) as mock_run,
        ):
            result = cli_runner.invoke(cli_app, ["multirun", "run", str(multirun_config_path)])

        assert result.exit_code == 0, result.output
        mock_load.assert_called_once_with(multirun_config_path, run_type="multirun")
        mock_run.assert_called_once_with(mock_multirun_job, mlflow=False)

    def test_run_forwards_mlflow_flag(
        self,
        cli_runner: CliRunner,
        multirun_config_path: Path,
        mock_multirun_job: SimpleNamespace,
        mock_multirun_result: MultiRunResult,
    ) -> None:
        with (
            patch(
                "dlkit.interfaces.cli.commands.multirun.load_config",
                return_value=mock_multirun_job,
            ),
            patch(
                "dlkit.interfaces.cli.commands.multirun.api_run_multirun_config",
                return_value=mock_multirun_result,
            ) as mock_run,
        ):
            result = cli_runner.invoke(
                cli_app, ["multirun", "run", "--mlflow", str(multirun_config_path)]
            )

        assert result.exit_code == 0, result.output
        mock_run.assert_called_once_with(mock_multirun_job, mlflow=True)

    def test_run_reports_dlkit_errors_cleanly(
        self,
        cli_runner: CliRunner,
        multirun_config_path: Path,
        mock_multirun_job: SimpleNamespace,
    ) -> None:
        from dlkit.common.errors import WorkflowError

        with (
            patch(
                "dlkit.interfaces.cli.commands.multirun.load_config",
                return_value=mock_multirun_job,
            ),
            patch(
                "dlkit.interfaces.cli.commands.multirun.api_run_multirun_config",
                side_effect=WorkflowError("sweep failed", {"workflow": "multirun"}),
            ),
        ):
            result = cli_runner.invoke(cli_app, ["multirun", "run", str(multirun_config_path)])

        assert result.exit_code == 1


class TestMultirunValidateCommand:
    """`dlkit multirun validate CONFIG.toml` — dry run, no execution."""

    def test_validate_expands_children_without_executing(
        self,
        cli_runner: CliRunner,
        multirun_config_path: Path,
        mock_multirun_job: SimpleNamespace,
    ) -> None:
        source_a = ExplicitFileSource(id="a", label="a", files=(Path("a.toml"),))
        source_b = ExplicitFileSource(id="b", label="b", files=(Path("b.toml"),))
        # `settings` is never touched by validate_sweep (only id/label/run_name
        # are printed), so a bare Mock() stands in for real WorkflowSettings.
        spec_a = RunSpec(id="a", label="a", settings=Mock(), run_name="a")
        spec_b = RunSpec(id="b", label="b", settings=Mock(), run_name="b")

        with (
            patch(
                "dlkit.interfaces.cli.commands.multirun.load_config",
                return_value=mock_multirun_job,
            ) as mock_load,
            patch(
                "dlkit.interfaces.cli.commands.multirun.build_child_sources",
                return_value=(source_a, source_b),
            ) as mock_build,
            patch(
                "dlkit.interfaces.cli.commands.multirun.expand_child_sources",
                return_value=(spec_a, spec_b),
            ) as mock_expand,
            patch("dlkit.interfaces.cli.commands.multirun.api_run_multirun_config") as mock_run,
        ):
            result = cli_runner.invoke(cli_app, ["multirun", "validate", str(multirun_config_path)])

        assert result.exit_code == 0, result.output
        mock_load.assert_called_once_with(multirun_config_path, run_type="multirun")
        mock_build.assert_called_once_with(mock_multirun_job.runs)
        mock_expand.assert_called_once_with((source_a, source_b))
        mock_run.assert_not_called()

    def test_validate_surfaces_expansion_errors(
        self,
        cli_runner: CliRunner,
        multirun_config_path: Path,
        mock_multirun_job: SimpleNamespace,
    ) -> None:
        from dlkit.common.errors import ConfigValidationError

        with (
            patch(
                "dlkit.interfaces.cli.commands.multirun.load_config",
                return_value=mock_multirun_job,
            ),
            patch(
                "dlkit.interfaces.cli.commands.multirun.build_child_sources",
                return_value=(),
            ),
            patch(
                "dlkit.interfaces.cli.commands.multirun.expand_child_sources",
                side_effect=ConfigValidationError("Duplicate multirun child id: 'a'"),
            ),
        ):
            result = cli_runner.invoke(cli_app, ["multirun", "validate", str(multirun_config_path)])

        assert result.exit_code == 1
