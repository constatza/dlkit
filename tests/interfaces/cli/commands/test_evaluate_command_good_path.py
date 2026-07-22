"""Tests for the evaluate command's checkpoint-selection flag wiring.

Mocks `evaluate_api` (the underlying `dlkit.interfaces.inference.evaluate`
call) to isolate CLI-level flag translation from the real evaluation
pipeline — real-run coverage for `--run-id`/`--latest-run` resolution itself
lives in `tests/integration/test_evaluate_cli_integration.py`.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from typer.testing import CliRunner

from dlkit.common.checkpoint_source import LatestRunCheckpoint, RunCheckpoint
from dlkit.interfaces.cli.commands.evaluate import app as evaluate_app


def _make_mock_evaluation_result() -> MagicMock:
    """Create a mock EvaluationResult with the fields the presenter reads."""
    mock_result = MagicMock()
    mock_result.duration_seconds = 0.1
    mock_result.figures = {}
    mock_result.metrics = {"mae": 0.1, "rmse": 0.2, "r2": 0.9}
    mock_result.mlflow_run_id = None
    return mock_result


class TestEvaluateCommandCheckpointSelection:
    @patch("dlkit.interfaces.cli.commands.evaluate.load_config")
    @patch("dlkit.interfaces.cli.commands.evaluate.evaluate_api")
    @patch("dlkit.interfaces.cli.commands.evaluate.present_evaluation_result")
    def test_evaluate_with_positional_checkpoint_passes_checkpoint_path(
        self,
        mock_present_result: Mock,
        mock_evaluate_api: Mock,
        mock_load_config: Mock,
        cli_runner: CliRunner,
        sample_config_path: Path,
        sample_checkpoint_path: Path,
        sample_settings: object,
    ) -> None:
        mock_load_config.return_value = sample_settings
        mock_evaluate_api.return_value = _make_mock_evaluation_result()

        result = cli_runner.invoke(
            evaluate_app, [str(sample_config_path), str(sample_checkpoint_path)]
        )

        assert result.exit_code == 0
        _, kwargs = mock_evaluate_api.call_args
        assert kwargs["checkpoint_path"] == sample_checkpoint_path
        assert kwargs["run_checkpoint"] is None
        mock_present_result.assert_called_once()

    @patch("dlkit.interfaces.cli.commands.evaluate.load_config")
    @patch("dlkit.interfaces.cli.commands.evaluate.evaluate_api")
    @patch("dlkit.interfaces.cli.commands.evaluate.present_evaluation_result")
    def test_evaluate_with_run_id_passes_run_checkpoint(
        self,
        mock_present_result: Mock,
        mock_evaluate_api: Mock,
        mock_load_config: Mock,
        cli_runner: CliRunner,
        sample_config_path: Path,
        sample_settings: object,
    ) -> None:
        mock_load_config.return_value = sample_settings
        mock_evaluate_api.return_value = _make_mock_evaluation_result()

        result = cli_runner.invoke(evaluate_app, [str(sample_config_path), "--run-id", "abc123"])

        assert result.exit_code == 0
        _, kwargs = mock_evaluate_api.call_args
        assert kwargs["checkpoint_path"] is None
        assert kwargs["run_checkpoint"] == RunCheckpoint(run_id="abc123")

    @patch("dlkit.interfaces.cli.commands.evaluate.load_config")
    @patch("dlkit.interfaces.cli.commands.evaluate.evaluate_api")
    @patch("dlkit.interfaces.cli.commands.evaluate.present_evaluation_result")
    def test_evaluate_with_latest_run_passes_latest_run_checkpoint(
        self,
        mock_present_result: Mock,
        mock_evaluate_api: Mock,
        mock_load_config: Mock,
        cli_runner: CliRunner,
        sample_config_path: Path,
        sample_settings: object,
    ) -> None:
        mock_load_config.return_value = sample_settings
        mock_evaluate_api.return_value = _make_mock_evaluation_result()

        result = cli_runner.invoke(evaluate_app, [str(sample_config_path), "--latest-run"])

        assert result.exit_code == 0
        _, kwargs = mock_evaluate_api.call_args
        assert kwargs["checkpoint_path"] is None
        assert kwargs["run_checkpoint"] == LatestRunCheckpoint()

    @patch("dlkit.interfaces.cli.commands.evaluate.load_config")
    def test_evaluate_with_run_id_and_latest_run_together_fails_cleanly(
        self,
        mock_load_config: Mock,
        cli_runner: CliRunner,
        sample_config_path: Path,
        sample_settings: object,
    ) -> None:
        mock_load_config.return_value = sample_settings

        result = cli_runner.invoke(
            evaluate_app, [str(sample_config_path), "--run-id", "abc123", "--latest-run"]
        )

        assert result.exit_code == 1
        assert "Pass at most one checkpoint-selection method" in result.output
        assert "Traceback" not in result.output

    @patch("dlkit.interfaces.cli.commands.evaluate.load_config")
    def test_evaluate_with_checkpoint_and_run_id_together_fails_cleanly(
        self,
        mock_load_config: Mock,
        cli_runner: CliRunner,
        sample_config_path: Path,
        sample_checkpoint_path: Path,
        sample_settings: object,
    ) -> None:
        mock_load_config.return_value = sample_settings

        result = cli_runner.invoke(
            evaluate_app,
            [str(sample_config_path), str(sample_checkpoint_path), "--run-id", "abc123"],
        )

        assert result.exit_code == 1
        assert "Pass at most one checkpoint-selection method" in result.output
        assert "Traceback" not in result.output
