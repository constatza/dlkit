from __future__ import annotations

import numpy as np
import torch

from dlkit.core.postprocessing import is_graph_output, to_plot_data
from dlkit.core.postprocessing import to_plot_graph_data


def test_is_graph_output_with_dict():
    g = {
        "x": torch.randn(4, 3),
        "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]]),
    }
    assert is_graph_output(g) is True


def test_to_plot_graph_data_from_dicts():
    graphs = [
        {
            "x": torch.randn(3, 2),
            "edge_index": torch.tensor([[0, 1], [1, 2]]),
            "y": torch.tensor([1, 0, 1]),
        },
        {
            "x": torch.randn(2, 2),
            "edge_index": torch.tensor([[0], [1]]),
        },
    ]
    out = to_plot_graph_data(graphs)
    assert isinstance(out, list)
    assert len(out) == 2
    assert isinstance(out[0]["nodes"], np.ndarray)
    assert isinstance(out[0]["edges"], np.ndarray)
    assert out[0]["nodes"].shape[0] == 3
    assert out[1]["nodes"].shape[0] == 2


def test_to_plot_data_facade_graph():
    graphs = [
        {
            "x": torch.randn(3, 2),
            "edge_index": torch.tensor([[0, 1], [1, 2]]),
        },
        {
            "x": torch.randn(2, 2),
            "edge_index": torch.tensor([[0], [1]]),
        },
    ]
    out = to_plot_data(graphs)
    assert isinstance(out, list)
    assert "nodes" in out[0] and "edges" in out[0]
