"""Graph prediction presentation utilities for the CLI."""

from __future__ import annotations

from typing import Any

from rich.pretty import Pretty

from .array import to_numpy
from .presenter_utils import is_graph_output


def to_plot_data(preds: Any, targets: Any | None = None) -> list[dict[str, Any]] | dict[str, Any]:
    """Convert graph predictions into plot-friendly dictionaries."""

    if isinstance(preds, list):
        result = [_extract_graph(item) for item in preds]
        if targets is not None:
            if isinstance(targets, list) and len(targets) == len(result):
                for index, target in enumerate(targets):
                    result[index]["targets"] = _as_numpy_or_none(target)
            else:
                for item in result:
                    item["targets"] = _as_numpy_or_none(targets)
        return result

    item = _extract_graph(preds)
    if targets is not None:
        item["targets"] = _as_numpy_or_none(targets)
    return item


def _extract_graph(obj: Any) -> dict[str, Any]:
    """Extract a plot-friendly dictionary from a graph-like object."""

    if not is_graph_output(obj):  # pragma: no cover - defensive
        return {"nodes": None, "edges": None}

    return {
        "nodes": _as_numpy_or_none(_get_attr(obj, "x")),
        "edges": _as_numpy_or_none(_get_attr(obj, "edge_index")),
        "edge_attrs": _as_numpy_or_none(_get_attr(obj, "edge_attr")),
        "preds": _as_numpy_or_none(
            _get_attr(obj, "pred")
            or _get_attr(obj, "y_hat")
            or _get_attr(obj, "logits")
            or _get_attr(obj, "out")
        ),
        "targets": _as_numpy_or_none(_get_attr(obj, "y")),
    }


def _get_attr(obj: Any, name: str) -> Any | None:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _as_numpy_or_none(value: Any) -> Any:
    if value is None:
        return None
    return to_numpy(value)


class GraphResultPresenter:
    """Simple graph-result presenter implementing the CLI presenter protocol."""

    def present(self, result: Any, console) -> None:
        console.print(Pretty(to_plot_data(result)))
