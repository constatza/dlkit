from __future__ import annotations

import numpy as np
import torch

from dlkit.core.postprocessing import (
    stack_batches,
    to_numpy,
    to_plot_array_data,
    to_plot_data,
)


def _randn(*shape: int) -> torch.Tensor:
    return torch.randn(*shape)


def test_stack_batches_equal_shapes_stack_mode():
    b1 = _randn(3, 5)
    b2 = _randn(2, 5)
    out = stack_batches([b1, b2], mode="stack")
    assert isinstance(out, np.ndarray)
    assert out.shape == (5, 5)


def test_stack_batches_ragged_list_mode():
    b1 = _randn(3, 4)
    b2 = _randn(5, 4)
    # stack in list mode should return list of arrays
    out = stack_batches([b1, b2], mode="list")
    assert isinstance(out, list)
    assert all(isinstance(x, np.ndarray) for x in out)
    assert out[0].shape == (3, 4)
    assert out[1].shape == (5, 4)


def test_stack_batches_dict_of_tensors():
    batches = [
        {"logits": _randn(2, 3), "y": torch.randint(0, 3, (2,))},
        {"logits": _randn(1, 3), "y": torch.randint(0, 3, (1,))},
    ]
    out = stack_batches(batches, mode="stack")
    assert isinstance(out, dict)
    assert out["logits"].shape == (3, 3)
    assert out["y"].shape == (3,)


def test_to_numpy_nested():
    data = {"a": torch.tensor([1, 2]), "b": [torch.tensor([3.0]), 4]}
    arr = to_numpy(data)
    assert isinstance(arr, dict)
    assert isinstance(arr["a"], np.ndarray)
    assert isinstance(arr["b"], list)
    assert isinstance(arr["b"][0], np.ndarray)


def test_to_plot_array_data_dict():
    preds = [
        {"logits": _randn(2, 2), "y": torch.tensor([0, 1])},
        {"logits": _randn(1, 2), "y": torch.tensor([1])},
    ]
    out = to_plot_array_data(preds)
    assert isinstance(out, dict)
    assert set(out.keys()) >= {"logits", "y"}
    assert out["logits"].shape[0] == 3
    assert out["y"].shape == (3,)


def test_to_plot_data_facade_array():
    preds = [_randn(2, 3), _randn(1, 3)]
    out = to_plot_data(preds)
    assert isinstance(out, np.ndarray)
    assert out.shape == (3, 3)
