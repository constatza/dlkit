"""Graph postprocessing utilities.

This module prepares graph predictions for writing or plotting, without
forcing variable-sized graphs to a single stacked array.
"""

from __future__ import annotations

from typing import Any

from .array import to_numpy
from .core import is_graph_output


def _get_attr(obj: Any, name: str) -> Any | None:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _as_numpy_or_none(x: Any) -> Any:
    if x is None:
        return None
    return to_numpy(x)


def _extract_graph(obj: Any) -> dict[str, Any]:
    """Extract a plot-friendly dictionary from a graph-like object.

    Keys in result:
    - nodes: node features array or None
    - edges: edge index [2, E] as numpy
    - node_attrs: optional node-level predictions/targets if distinguishable
    - edge_attrs: optional edge attributes
    - preds: heuristic prediction tensor on nodes/graph
    - targets: heuristic targets if present ("y")
    """
    if not is_graph_output(obj):  # pragma: no cover - defensive
        return {"nodes": None, "edges": None}

    nodes = _get_attr(obj, "x")
    edges = _get_attr(obj, "edge_index")
    edge_attr = _get_attr(obj, "edge_attr")
    targets = _get_attr(obj, "y")

    # Heuristic for predictions
    preds = (
        _get_attr(obj, "pred")
        or _get_attr(obj, "y_hat")
        or _get_attr(obj, "logits")
        or _get_attr(obj, "out")
    )

    return {
        "nodes": _as_numpy_or_none(nodes),
        "edges": _as_numpy_or_none(edges),
        "edge_attrs": _as_numpy_or_none(edge_attr),
        "preds": _as_numpy_or_none(preds),
        "targets": _as_numpy_or_none(targets),
    }


def to_plot_data(preds: Any, targets: Any | None = None) -> list[dict[str, Any]] | dict[str, Any]:
    """Convert graph predictions to plot-friendly dictionaries.

    - Input can be a single graph object or a list of them.
    - Returns a list of dicts with numpy arrays.
    - If targets are provided separately and align by position, they are
      attached to each item under the "targets" key.
    """
    if isinstance(preds, list):
        result = [_extract_graph(p) for p in preds]
        if targets is not None:
            if isinstance(targets, list) and len(targets) == len(result):
                for i, t in enumerate(targets):
                    result[i]["targets"] = _as_numpy_or_none(t)
            else:  # broadcast single target to all
                for item in result:
                    item["targets"] = _as_numpy_or_none(targets)
        return result

    # Single graph
    item = _extract_graph(preds)
    if targets is not None:
        item["targets"] = _as_numpy_or_none(targets)
    return item
