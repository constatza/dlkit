"""CLI presentation helpers for formatting and postprocessing results."""

from typing import Any

from .array import ArrayResultPresenter, stack_batches, to_numpy
from .array import to_plot_data as to_plot_array_data
from .graph import GraphResultPresenter
from .graph import to_plot_data as to_plot_graph_data
from .presenter_utils import is_graph_output, summarize
from .protocol import IResultPresenter


def to_plot_data(preds: Any, targets: Any | None = None) -> Any:
    """Route prediction data to array or graph presentation helpers."""

    if isinstance(preds, list) and preds:
        if any(is_graph_output(item) for item in preds):
            return to_plot_graph_data(preds, targets)
        return to_plot_array_data(preds, targets)
    if is_graph_output(preds):
        return to_plot_graph_data(preds, targets)
    return to_plot_array_data(preds, targets)


__all__ = [
    "ArrayResultPresenter",
    "GraphResultPresenter",
    "IResultPresenter",
    "is_graph_output",
    "stack_batches",
    "summarize",
    "to_numpy",
    "to_plot_array_data",
    "to_plot_data",
    "to_plot_graph_data",
]
