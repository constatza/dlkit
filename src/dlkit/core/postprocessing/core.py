"""Core types and helpers for postprocessing.

This module contains minimal shared utilities and error types used by
array and graph postprocessing. It intentionally keeps logic small to
adhere to KISS and SOLID (no heavy central dispatcher).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NotStackableError(Exception):
    """Raised when batches cannot be stacked in strict mode."""

    message: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


def is_graph_output(obj: Any) -> bool:
    """Heuristically detect whether an object represents a graph prediction.

    Rules (any true → graph):
    - Mapping with key "edge_index"
    - Object with attribute "edge_index"
    - Torch Geometric Data-like (duck-typed via attributes)

    Args:
        obj: Object to inspect

    Returns:
        True if it appears to be a graph-like object.
    """
    try:
        # Mapping-like
        if isinstance(obj, dict) and "edge_index" in obj:
            return True
        # Attr-based (torch_geometric.Data style)
        if hasattr(obj, "edge_index"):
            return True
    except Exception:
        return False
    return False


def summarize(obj: Any) -> dict[str, Any]:
    """Summarize predictions for quick inspection and logging.

    Dense:
        - Count, element type, first shape
    Graph:
        - Number of graphs, total nodes/edges (best-effort), and per-graph sizes

    Args:
        obj: Predictions or batches.

    Returns:
        Summary dictionary with lightweight metadata.
    """
    summary: dict[str, Any] = {}

    # List/sequence case
    if isinstance(obj, list):
        summary["count"] = len(obj)
        if not obj:
            return summary
        first = obj[0]
        summary["type"] = type(first).__name__
        if is_graph_output(first):
            sizes: list[tuple[int | None, int | None]] = []
            total_nodes = 0
            total_edges = 0
            for item in obj:
                n, e = _graph_sizes(item)
                sizes.append((n, e))
                total_nodes += int(n or 0)
                total_edges += int(e or 0)
            summary["graphs"] = {
                "total": len(obj),
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "sizes": sizes,
            }
        else:
            try:
                import torch  # lazy

                if isinstance(first, torch.Tensor):
                    summary["shape"] = tuple(first.shape)
                elif isinstance(first, dict):
                    summary["keys"] = list(first.keys())
            except Exception:
                pass
        return summary

    # Single graph
    if is_graph_output(obj):
        n, e = _graph_sizes(obj)
        summary["graphs"] = {"total": 1, "total_nodes": n, "total_edges": e, "sizes": [(n, e)]}
        return summary

    # Dense single object
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            summary["shape"] = tuple(obj.shape)
            summary["dtype"] = str(obj.dtype)
    except Exception:
        pass
    return summary


def _graph_sizes(obj: Any) -> tuple[int | None, int | None]:
    """Best-effort graph size extraction (num_nodes, num_edges)."""
    try:
        import torch

        if isinstance(obj, dict):
            num_nodes = int(obj.get("num_nodes") or (obj.get("x").shape[0] if "x" in obj else 0))
            edge_index = obj.get("edge_index")
            if isinstance(edge_index, torch.Tensor):
                num_edges = int(edge_index.shape[1])
            else:
                try:
                    num_edges = int(edge_index.shape[1])  # type: ignore[attr-defined]
                except Exception:
                    num_edges = None
            return num_nodes, num_edges

        # Attr case
        if hasattr(obj, "num_nodes") and getattr(obj, "num_nodes") is not None:
            n = int(getattr(obj, "num_nodes"))
        elif hasattr(obj, "x") and getattr(obj, "x") is not None:
            x = getattr(obj, "x")
            n = int(getattr(x, "shape")[0])
        else:
            n = None

        ei = getattr(obj, "edge_index", None)
        if ei is not None:
            num_edges = int(getattr(ei, "shape")[1])
        else:
            num_edges = None
        return n, num_edges
    except Exception:
        return None, None
