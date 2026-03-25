"""Postprocessing utilities for inference outputs.

This package provides small, focused functions to:

- Stack dense batch outputs into arrays
- Convert tensors recursively to numpy
- Prepare plot-friendly dataflow for dense and graph predictions

Design goals:
- KISS: function-based API, minimal state
- SOLID: separate array vs graph concerns, open for extension
"""

from typing import Any

from .array import stack_batches, to_numpy
from .array import to_plot_data as to_plot_array_data
from .core import is_graph_output, summarize
from .graph import to_plot_data as to_plot_graph_data


def to_plot_data(preds: Any, targets: Any | None = None) -> Any:
    """Facade that routes to array or graph conversion.

    If predictions look graph-like (list with graph elements or single graph),
    dispatch to graph conversion; otherwise use array conversion.
    """
    if isinstance(preds, list) and preds:
        if any(is_graph_output(p) for p in preds):
            return to_plot_graph_data(preds, targets)
        return to_plot_array_data(preds, targets)
    if is_graph_output(preds):
        return to_plot_graph_data(preds, targets)
    return to_plot_array_data(preds, targets)


__all__ = [
    "is_graph_output",
    "stack_batches",
    "summarize",
    "to_numpy",
    "to_plot_array_data",
    "to_plot_data",
    "to_plot_graph_data",
]
