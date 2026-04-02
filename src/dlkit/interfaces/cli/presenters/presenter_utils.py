"""Shared helpers for CLI prediction presentation."""

from __future__ import annotations

from typing import Any


class NotStackableError(Exception):
    """Raised when dense batches cannot be stacked in strict mode."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


def is_graph_output(obj: Any) -> bool:
    """Heuristically detect whether an object represents a graph prediction."""

    try:
        if isinstance(obj, dict) and "edge_index" in obj:
            return True
        if hasattr(obj, "edge_index"):
            return True
    except Exception:
        return False
    return False


def summarize(obj: Any) -> dict[str, Any]:
    """Summarize predictions for quick CLI inspection and logging."""

    summary: dict[str, Any] = {}

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
                num_nodes, num_edges = _graph_sizes(item)
                sizes.append((num_nodes, num_edges))
                total_nodes += int(num_nodes or 0)
                total_edges += int(num_edges or 0)
            summary["graphs"] = {
                "total": len(obj),
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "sizes": sizes,
            }
        else:
            try:
                import torch

                if isinstance(first, torch.Tensor):
                    summary["shape"] = tuple(first.shape)
                elif isinstance(first, dict):
                    summary["keys"] = list(first.keys())
            except Exception:
                pass
        return summary

    if is_graph_output(obj):
        num_nodes, num_edges = _graph_sizes(obj)
        summary["graphs"] = {
            "total": 1,
            "total_nodes": num_nodes,
            "total_edges": num_edges,
            "sizes": [(num_nodes, num_edges)],
        }
        return summary

    try:
        import torch

        if isinstance(obj, torch.Tensor):
            summary["shape"] = tuple(obj.shape)
            summary["dtype"] = str(obj.dtype)
    except Exception:
        pass
    return summary


def _graph_sizes(obj: Any) -> tuple[int | None, int | None]:
    """Extract graph node and edge counts on a best-effort basis."""

    try:
        if isinstance(obj, dict):
            nodes = obj.get("x")
            num_nodes = int(obj.get("num_nodes") or (nodes.shape[0] if nodes is not None else 0))
            edge_index = obj.get("edge_index")
            try:
                num_edges = int(edge_index.shape[1])
            except Exception:
                num_edges = None
            return num_nodes, num_edges

        if hasattr(obj, "num_nodes") and obj.num_nodes is not None:
            num_nodes = int(obj.num_nodes)
        elif hasattr(obj, "x") and obj.x is not None:
            num_nodes = int(obj.x.shape[0])
        else:
            num_nodes = None

        edge_index = getattr(obj, "edge_index", None)
        if edge_index is not None:
            num_edges = int(edge_index.shape[1])
        else:
            num_edges = None
        return num_nodes, num_edges
    except Exception:
        return None, None
