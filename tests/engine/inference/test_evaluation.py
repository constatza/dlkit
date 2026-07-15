"""Tests for eval-only orchestration: metrics, figures, and full checkpoint evaluation."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import pytest
import torch
from matplotlib.figure import Figure

from dlkit.engine.inference.evaluation import (
    compute_regression_metrics,
    evaluate_checkpoint,
    generate_regression_figures,
    log_evaluation_result,
)
from dlkit.infrastructure.config.plot_settings import PlotSettings

N_SAMPLES = 20


@pytest.fixture
def perfect_predictions() -> tuple[torch.Tensor, torch.Tensor]:
    """Predictions identical to targets — deterministic MAE=0, RMSE=0, R2=1."""
    targets = torch.linspace(0.0, 1.0, N_SAMPLES)
    return targets.clone(), targets


@pytest.fixture
def plots_all_enabled() -> PlotSettings:
    return PlotSettings(
        enabled=True, parity=True, residual=True, error_histogram=True, residual_vs_index=True
    )


@pytest.fixture
def varied_predictor_and_datamodule(
    make_predictor: Callable[..., MagicMock],
    make_eval_datamodule: Callable[..., MagicMock],
) -> tuple[MagicMock, MagicMock]:
    """Predictor/datamodule pair with non-degenerate (non-constant) data.

    Constant-valued batches make KDE/correlation computations in the figure
    generators divide by a zero standard deviation; varied data avoids that.
    """
    targets = torch.linspace(0.0, 1.0, N_SAMPLES).unsqueeze(1)
    predictions = targets + 0.1
    predictor = make_predictor(predictions=[predictions])
    datamodule = make_eval_datamodule(batches=[({"x": targets}, {"y": targets})])
    return predictor, datamodule


def test_compute_regression_metrics_perfect_predictions(
    perfect_predictions: tuple[torch.Tensor, torch.Tensor],
) -> None:
    predictions, targets = perfect_predictions

    metrics = compute_regression_metrics(predictions, targets)

    assert metrics.keys() == {"mae", "rmse", "r2"}
    assert metrics["mae"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["r2"] == pytest.approx(1.0, abs=1e-6)


def test_compute_regression_metrics_constant_offset() -> None:
    targets = torch.zeros(N_SAMPLES)
    predictions = torch.ones(N_SAMPLES)

    metrics = compute_regression_metrics(predictions, targets)

    assert metrics["mae"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["rmse"] == pytest.approx(1.0, abs=1e-6)


def test_generate_regression_figures_respects_plot_settings_flags(
    perfect_predictions: tuple[torch.Tensor, torch.Tensor],
) -> None:
    predictions, targets = perfect_predictions
    plots = PlotSettings(enabled=True, parity=True)

    figures = generate_regression_figures(predictions, targets, plots)

    try:
        assert figures.keys() == {"parity_plot"}
        assert isinstance(figures["parity_plot"], Figure)
    finally:
        for fig in figures.values():
            plt.close(fig)


def test_generate_regression_figures_empty_when_no_plots_enabled(
    perfect_predictions: tuple[torch.Tensor, torch.Tensor],
) -> None:
    predictions, targets = perfect_predictions

    figures = generate_regression_figures(predictions, targets, PlotSettings())

    assert figures == {}


def test_evaluate_checkpoint_returns_predictions_targets_metrics_and_figures(
    varied_predictor_and_datamodule: tuple[MagicMock, MagicMock],
    plots_all_enabled: PlotSettings,
) -> None:
    predictor, datamodule = varied_predictor_and_datamodule

    result = evaluate_checkpoint(predictor, datamodule, plots_all_enabled)

    try:
        assert result.predictions.shape == (N_SAMPLES, 1)
        assert result.targets.shape == (N_SAMPLES, 1)
        assert result.metrics["mae"] == pytest.approx(0.1, abs=1e-6)
        assert set(result.figures) == {
            "parity_plot",
            "residual_plot",
            "error_histogram",
            "residual_vs_index",
        }
        assert result.duration_seconds >= 0.0
    finally:
        for fig in result.figures.values():
            plt.close(fig)


def test_log_evaluation_result_logs_metrics_and_artifacts_without_closing_figures(
    varied_predictor_and_datamodule: tuple[MagicMock, MagicMock],
    plots_all_enabled: PlotSettings,
) -> None:
    predictor, datamodule = varied_predictor_and_datamodule
    result = evaluate_checkpoint(predictor, datamodule, plots_all_enabled)
    run_context = MagicMock()

    try:
        log_evaluation_result(result, run_context, plots_all_enabled)

        run_context.log_metrics.assert_called_once_with(result.metrics)
        assert run_context.log_artifact_content.call_count == len(result.figures)
        for fig in result.figures.values():
            assert plt.fignum_exists(fig.number)
    finally:
        for fig in result.figures.values():
            plt.close(fig)
