# test_subsample_simple.py

import pytest
import torch
from dlkit.transforms.subset import TensorSubset


# -------------------------------
# Fixtures
# -------------------------------


@pytest.fixture
def case_1d():
    """
    1D Tensor: [1, 2, 3, 4]
    dim = 0
    keep = [0, 2]
    drop = [1]
    - Valid range along dim 0: [0, 1, 2, 3]
    - Exclude index 1 → available = [0, 2, 3]
    - Intersect with keep [0, 2] → final = [0, 2]
    Expected output: tensor([1, 3])
    """
    tensor = torch.tensor([1, 2, 3, 4])
    dim = 0
    keep = [0, 2]
    expected = torch.tensor([1, 3])  # indices 0 and 2
    return tensor, dim, keep, expected


@pytest.fixture
def case_2d():
    """
    2D Tensor (2×4):
        [[ 5,  6,  7,  8],
         [ 9, 10, 11, 12]]
    dim = 1
    keep = [0, 3]
    drop = [1]
    - Valid range along dim 1: [0, 1, 2, 3]
    - Exclude index 1 → available = [0, 2, 3]
    - Intersect with keep [0, 3] → final = [0, 3]
    Expected output: tensor([[ 5,  8],
                            [ 9, 12]])
    """
    tensor = torch.tensor([[5, 6, 7, 8], [9, 10, 11, 12]])
    dim = 1
    keep = [0, 3]
    expected = torch.tensor([[5, 8], [9, 12]])  # keep columns 0 and 3
    return tensor, dim, keep, expected


@pytest.fixture
def case_3d():
    """
    3D Tensor (2×3×3):
    [
      [[ 0,  1,  2],
       [ 3,  4,  5],
       [ 6,  7,  8]],

      [[ 9, 10, 11],
       [12, 13, 14],
       [15, 16, 17]]
    ]
    dim = 2
    keep = [0, 2]
    drop = [1]
    - Valid range along dim 2: [0, 1, 2]
    - Exclude index 1 → available = [0, 2]
    - Intersect with keep [0, 2] → final = [0, 2]
    Expected output: tensor([
        [[ 0,  2],
         [ 3,  5],
         [ 6,  8]],

        [[ 9, 11],
         [12, 14],
         [15, 17]]
    ])  # shape (2×3×2)
    """
    tensor = torch.tensor([
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        [[9, 10, 11], [12, 13, 14], [15, 16, 17]],
    ])
    dim = 2
    keep = [0, 2]
    expected = torch.tensor([
        [[0, 2], [3, 5], [6, 8]],
        [[9, 11], [12, 14], [15, 17]],
    ])  # select indices 0 and 2 along dim=2
    return tensor, dim, keep, expected


@pytest.fixture
def case_1d_with_slice():
    tensor = torch.arange(15)
    expected = torch.arange(0, 10, 2)
    dim = 0
    keep = slice(0, 10, 2)
    return tensor, dim, keep, expected


# -------------------------------
# Three Test Functions
# -------------------------------


def test_subsample_1d(case_1d):
    """
    Test 1D subsampling:
      Input: [1, 2, 3, 4], dim=0, keep=[0,2], drop=[1] → output [1, 3]
    """
    tensor, dim, keep, expected = case_1d
    layer = TensorSubset(
        dim=dim,
        keep=keep,
        input_shape=tensor.shape,
    )
    output = layer(tensor)
    assert torch.equal(output, expected), f"1D: got {output.tolist()}, expected {expected.tolist()}"


def test_subsample_2d(case_2d):
    """
    Test 2D subsampling:
      Input (2×4), dim=1, keep=[0,3], drop=[1] → output [[5,8],[9,12]]
    """
    tensor, dim, keep, expected = case_2d
    layer = TensorSubset(dim=dim, keep=keep, input_shape=tensor.shape)
    output = layer(tensor)
    assert torch.equal(output, expected), f"2D: got {output.tolist()}, expected {expected.tolist()}"


def test_subsample_3d(case_3d):
    """
    Test 3D subsampling:
      Input (2×3×3), dim=2, keep=[0,2], drop=[1] → output shape (2×3×2)
    """
    tensor, dim, keep, expected = case_3d
    layer = TensorSubset(
        dim=dim,
        keep=keep,
        input_shape=tensor.shape,
    )
    output = layer(tensor)
    assert torch.equal(output, expected), f"3D: got {output.tolist()}, expected {expected.tolist()}"


def test_subsample_1d_with_slice(case_1d_with_slice):
    tensor, dim, keep, expected = case_1d_with_slice
    layer = TensorSubset(
        dim=dim,
        keep=keep,
        input_shape=tensor.shape,
    )
    output = layer(tensor)
    assert torch.equal(output, expected), f"1D: got {output.tolist()}, expected {expected.tolist()}"
