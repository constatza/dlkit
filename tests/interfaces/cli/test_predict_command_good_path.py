"""Tests for prediction command (stateful predictor-based inference with training configs)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from typer.testing import CliRunner

from dlkit.interfaces.cli.commands.predict import app as predict_app
from dlkit.interfaces.api.domain import ConfigurationError
from dlkit.tools.config import GeneralSettings


class TestPredictCommand:
    @patch("dlkit.interfaces.cli.commands.predict.load_config")
    @patch("dlkit.interfaces.cli.commands.predict.load_model")
    @patch("dlkit.interfaces.cli.commands.predict.present_inference_result")
    def test_predict_with_valid_inputs_succeeds(
        self,
        mock_present_result: Mock,
        mock_load_model: Mock,
        mock_load_config: Mock,
        cli_runner: CliRunner,
        sample_config_path: Path,
        sample_checkpoint_path: Path,
        sample_settings: GeneralSettings,
        mock_successful_inference_result,
    ) -> None:
        mock_load_config.return_value = sample_settings

        # Create a mock predictor that returns batch results
        mock_predictor = MagicMock()
        mock_predictor.predict_from_config.return_value = iter([mock_successful_inference_result])
        mock_load_model.return_value = mock_predictor

        result = cli_runner.invoke(
            predict_app, [str(sample_config_path), str(sample_checkpoint_path)]
        )

        assert result.exit_code == 0

        mock_load_config.assert_called_once()
        mock_load_model.assert_called_once()
        mock_predictor.predict_from_config.assert_called_once()
        mock_predictor.unload.assert_called_once()
        mock_present_result.assert_called_once()

    def test_infer_with_missing_checkpoint_fails(
        self,
        cli_runner: CliRunner,
        sample_config_path: Path,
        tmp_path: Path,
    ) -> None:
        missing_checkpoint = tmp_path / "missing.ckpt"

        result = cli_runner.invoke(predict_app, [str(sample_config_path), str(missing_checkpoint)])

        assert result.exit_code == 1

    @patch("dlkit.interfaces.cli.commands.predict.load_config")
    def test_infer_with_invalid_config_fails(
        self,
        mock_load_config: Mock,
        cli_runner: CliRunner,
        sample_config_path: Path,
        sample_checkpoint_path: Path,
        mock_configuration_error: ConfigurationError,
    ) -> None:
        mock_load_config.side_effect = mock_configuration_error

        result = cli_runner.invoke(
            predict_app, [str(sample_config_path), str(sample_checkpoint_path)]
        )

        assert result.exit_code == 1
        mock_load_config.assert_called_once()

    @patch("dlkit.interfaces.cli.commands.predict.load_config")
    @patch("dlkit.interfaces.cli.commands.predict.load_model")
    def test_infer_with_parameter_overrides(
        self,
        mock_load_model: Mock,
        mock_load_config: Mock,
        cli_runner: CliRunner,
        sample_config_path: Path,
        sample_checkpoint_path: Path,
        sample_settings: GeneralSettings,
        mock_successful_inference_result,
        tmp_path: Path,
    ) -> None:
        mock_load_config.return_value = sample_settings

        # Create a mock predictor that returns batch results
        mock_predictor = MagicMock()
        mock_predictor.predict_from_config.return_value = iter([mock_successful_inference_result])
        mock_load_model.return_value = mock_predictor

        output_dir = tmp_path / "custom_output"
        data_dir = tmp_path / "custom_data"

        result = cli_runner.invoke(
            predict_app,
            [
                str(sample_config_path),
                str(sample_checkpoint_path),
                "--output-dir",
                str(output_dir),
                "--dataflow-dir",
                str(data_dir),
                "--batch-size",
                "32",
            ],
        )

        assert result.exit_code == 0


class TestPredictMainCallback:
    def test_infer_without_args_shows_missing_argument_error(
        self,
        cli_runner: CliRunner,
    ) -> None:
        result = cli_runner.invoke(predict_app, [])

        assert result.exit_code == 2
        assert "Missing argument 'CONFIG_PATH'" in result.stderr

    def test_infer_without_checkpoint_shows_missing_argument_error(
        self,
        cli_runner: CliRunner,
        sample_config_path: Path,
    ) -> None:
        result = cli_runner.invoke(predict_app, [str(sample_config_path)])

        assert result.exit_code == 2
        assert "Missing argument 'CHECKPOINT'" in result.stderr

    @patch("dlkit.interfaces.cli.commands.predict._run_inference_impl")
    def test_infer_direct_invocation(
        self,
        mock_run_inference: Mock,
        cli_runner: CliRunner,
        sample_config_path: Path,
        sample_checkpoint_path: Path,
    ) -> None:
        result = cli_runner.invoke(
            predict_app, [str(sample_config_path), str(sample_checkpoint_path)]
        )

        assert result.exit_code == 0

        mock_run_inference.assert_called_once()
        args, kwargs = mock_run_inference.call_args
        assert kwargs["config_path"] == sample_config_path
        assert kwargs["checkpoint"] == sample_checkpoint_path

    @patch("dlkit.interfaces.cli.commands.predict._run_inference_impl")
    def test_infer_direct_invocation_with_options(
        self,
        mock_run_inference: Mock,
        cli_runner: CliRunner,
        sample_config_path: Path,
        sample_checkpoint_path: Path,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "outputs"

        result = cli_runner.invoke(
            predict_app,
            [
                str(sample_config_path),
                str(sample_checkpoint_path),
                "--output-dir",
                str(output_dir),
                "--batch-size",
                "64",
            ],
        )

        assert result.exit_code == 0

        mock_run_inference.assert_called_once()
        args, kwargs = mock_run_inference.call_args
        assert kwargs["output_dir"] == output_dir
        assert kwargs["batch_size"] == 64
        assert kwargs["save_predictions"] is True


class TestPredictHelperFunctions:
    @patch("dlkit.interfaces.cli.commands.predict.load_config")
    @patch("dlkit.interfaces.cli.commands.predict.load_model")
    @patch("dlkit.interfaces.cli.commands.predict.present_inference_result")
    def test_run_inference_impl_saves_predictions_by_default(
        self,
        mock_present_result: Mock,
        mock_load_model: Mock,
        mock_load_config: Mock,
        sample_config_path: Path,
        sample_checkpoint_path: Path,
        sample_settings: GeneralSettings,
        mock_successful_inference_result,
    ) -> None:
        from dlkit.interfaces.cli.commands.predict import _run_inference_impl

        mock_load_config.return_value = sample_settings

        # Create a mock predictor that returns batch results
        mock_predictor = MagicMock()
        mock_predictor.predict_from_config.return_value = iter([mock_successful_inference_result])
        mock_load_model.return_value = mock_predictor

        _run_inference_impl(config_path=sample_config_path, checkpoint=sample_checkpoint_path)

        mock_present_result.assert_called_once()
        args, kwargs = mock_present_result.call_args
        assert kwargs.get("save_predictions", True) is True

    @patch("dlkit.interfaces.cli.commands.predict.load_config")
    @patch("dlkit.interfaces.cli.commands.predict.load_model")
    @patch("dlkit.interfaces.cli.commands.predict.present_inference_result")
    def test_run_inference_impl_can_disable_prediction_saving(
        self,
        mock_present_result: Mock,
        mock_load_model: Mock,
        mock_load_config: Mock,
        sample_config_path: Path,
        sample_checkpoint_path: Path,
        sample_settings: GeneralSettings,
        mock_successful_inference_result,
    ) -> None:
        from dlkit.interfaces.cli.commands.predict import _run_inference_impl

        mock_load_config.return_value = sample_settings

        # Create a mock predictor that returns batch results
        mock_predictor = MagicMock()
        mock_predictor.predict_from_config.return_value = iter([mock_successful_inference_result])
        mock_load_model.return_value = mock_predictor

        _run_inference_impl(
            config_path=sample_config_path,
            checkpoint=sample_checkpoint_path,
            save_predictions=False,
        )

        mock_present_result.assert_called_once()
        args, kwargs = mock_present_result.call_args
        assert kwargs["save_predictions"] is False
