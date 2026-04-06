"""Result presentation adapter for CLI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from dlkit.common import InferenceResult, OptimizationResult, TrainingResult
from dlkit.interfaces.cli.presenters import summarize


def present_training_result(result: TrainingResult, console: Console) -> None:
    """Present training result in a formatted display.

    Args:
        result: Training result to display
        console: Rich console for output
    """
    # Create main results panel
    results_text = Text()
    results_text.append("🏋️ Training Results\n\n", style="bold blue")
    results_text.append(f"Duration: {result.duration_seconds:.2f} seconds\n", style="green")

    if result.metrics:
        results_text.append(f"Metrics: {len(result.metrics)} recorded\n", style="cyan")

    if result.artifacts:
        results_text.append(f"Artifacts: {len(result.artifacts)} saved\n", style="yellow")

    results_panel = Panel.fit(results_text, title="Training Summary", border_style="green")
    console.print(results_panel)

    # Display metrics table if available
    if result.metrics:
        _display_metrics_table(result.metrics, console, "Training Metrics")

    # Display artifacts table if available
    if result.artifacts:
        _display_artifacts_table(result.artifacts, console)


def present_inference_result(
    result: InferenceResult, console: Console, save_predictions: bool = True
) -> None:
    """Present inference result in a formatted display.

    Args:
        result: Inference result to display
        console: Rich console for output
        save_predictions: Whether predictions were saved
    """
    # Create main results panel
    results_text = Text()
    results_text.append("🔮 Inference Results\n\n", style="bold blue")
    results_text.append(f"Duration: {result.duration_seconds:.2f} seconds\n", style="green")

    if result.predictions is not None:
        results_text.append("Predictions: Generated successfully\n", style="cyan")
        if save_predictions:
            results_text.append("Predictions saved to output directory\n", style="yellow")

    if result.metrics:
        results_text.append(f"Metrics: {len(result.metrics)} recorded\n", style="cyan")

    results_panel = Panel.fit(results_text, title="Inference Summary", border_style="blue")
    console.print(results_panel)

    # Display metrics table if available
    if result.metrics:
        _display_metrics_table(result.metrics, console, "Inference Metrics")

    # Display prediction summary
    if result.predictions is not None:
        _display_prediction_summary(result.predictions, console)


def present_optimization_result(result: OptimizationResult, console: Console) -> None:
    """Present optimization result in a formatted display.

    Args:
        result: Optimization result to display
        console: Rich console for output
    """
    # Create main results panel
    results_text = Text()
    results_text.append("⚡ Optimization Results\n\n", style="bold blue")
    results_text.append(f"Duration: {result.duration_seconds:.2f} seconds\n", style="green")

    if result.best_trial:
        best_value = result.best_trial.get("value", "N/A")
        results_text.append(f"Best value: {best_value}\n", style="cyan")

        best_trial_num = result.best_trial.get("number", "N/A")
        results_text.append(f"Best trial: #{best_trial_num}\n", style="yellow")

    results_panel = Panel.fit(results_text, title="Optimization Summary", border_style="magenta")
    console.print(results_panel)

    # Display best parameters
    if result.best_trial and "params" in result.best_trial:
        _display_best_parameters(result.best_trial["params"], console)

    # Display study summary
    if result.study_summary:
        _display_study_summary(result.study_summary, console)

    # Display final training metrics
    if result.training_result.metrics:
        _display_metrics_table(result.training_result.metrics, console, "Final Training Metrics")


def _display_metrics_table(metrics: dict[str, Any], console: Console, title: str) -> None:
    """Display metrics in a formatted table.

    Args:
        metrics: Dictionary of metrics
        console: Rich console for output
        title: Table title
    """
    table = Table(title=title)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for metric_name, value in metrics.items():
        # Format value based on type
        if isinstance(value, float):
            value_str = f"{value:.6f}"
        else:
            value_str = str(value)

        table.add_row(metric_name, value_str)

    console.print(table)


def _display_artifacts_table(artifacts: dict[str, Path], console: Console) -> None:
    """Display artifacts in a formatted table.

    Args:
        artifacts: Dictionary of artifact paths
        console: Rich console for output
    """
    table = Table(title="Saved Artifacts")
    table.add_column("Artifact", style="cyan")
    table.add_column("Path", style="green")
    table.add_column("Exists", style="yellow")

    for artifact_name, path in artifacts.items():
        exists_str = "✅" if path.exists() else "❌"
        table.add_row(artifact_name, str(path), exists_str)

    console.print(table)


def _display_best_parameters(params: dict[str, Any], console: Console) -> None:
    """Display best parameters in a formatted table.

    Args:
        params: Dictionary of best parameters
        console: Rich console for output
    """
    table = Table(title="🏆 Best Parameters")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Type", style="yellow")

    for param_name, value in params.items():
        value_str = str(value)
        type_str = type(value).__name__
        table.add_row(param_name, value_str, type_str)

    console.print(table)


def _display_study_summary(summary: dict[str, Any], console: Console) -> None:
    """Display study summary in a formatted table.

    Args:
        summary: Dictionary of study summary information
        console: Rich console for output
    """
    table = Table(title="📊 Study Summary")
    table.add_column("Statistic", style="cyan")
    table.add_column("Value", style="green")

    for stat_name, value in summary.items():
        table.add_row(stat_name, str(value))

    console.print(table)


def _display_prediction_summary(predictions: Any, console: Console) -> None:
    """Display prediction summary.

    Args:
        predictions: Model predictions
        console: Rich console for output
    """
    # Create prediction summary based on type
    pred_text = Text("🔮 Prediction Summary\n", style="bold blue")

    if predictions is None:
        pred_text.append("No predictions generated", style="red")
    elif hasattr(predictions, "__len__"):
        pred_text.append(f"Generated {len(predictions)} predictions", style="green")
    else:
        pred_text.append("Predictions generated successfully", style="green")

    pred_panel = Panel.fit(pred_text, title="Predictions", border_style="blue")
    console.print(pred_panel)

    # Optional: concise postprocessing details (counts, shapes, keys, graph sizes)
    # Only shown when explicitly enabled via environment for verbose CLI output.
    if os.getenv("DLKIT_CLI_VERBOSE", "").lower() in {"1", "true", "yes", "on"}:
        try:
            info = summarize(predictions)
        except Exception:
            info = {}
        if info:
            table = Table(title="Prediction Details")
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="green")

            # Generic fields
            if "count" in info:
                table.add_row("count", str(info["count"]))
            if "type" in info:
                table.add_row("type", str(info["type"]))
            if "shape" in info:
                table.add_row("shape", str(tuple(info["shape"])))
            if "dtype" in info:
                table.add_row("dtype", str(info["dtype"]))
            if "keys" in info:
                table.add_row("keys", ", ".join(map(str, info["keys"])))

            # Graph specific
            graphs = info.get("graphs")
            if graphs:
                table.add_row("graphs.total", str(graphs.get("total")))
                if graphs.get("total_nodes") is not None:
                    table.add_row("graphs.total_nodes", str(graphs.get("total_nodes")))
                if graphs.get("total_edges") is not None:
                    table.add_row("graphs.total_edges", str(graphs.get("total_edges")))
                sizes = graphs.get("sizes")
                if sizes:
                    # Show up to first 3 sizes to stay concise
                    preview = ", ".join(str(s) for s in sizes[:3])
                    table.add_row("graphs.sizes[:3]", preview)

            console.print(table)
