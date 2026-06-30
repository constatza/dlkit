from __future__ import annotations

import unittest.mock

import torch
from torch.nn import ModuleList

from dlkit.domain.transforms.chain import TransformChain
from dlkit.domain.transforms.minmax import MinMaxScaler


def test_transform_chain_preserves_device_after_fit_from_dataloader() -> None:
    # Arrange
    chain = TransformChain(ModuleList([MinMaxScaler(dim=0)]), entry_name="x")
    mock_device = torch.device("cpu")

    dataloader = [torch.tensor([[0.0, 1.0], [2.0, 3.0]])]

    # Act
    with unittest.mock.patch.object(TransformChain, "to", wraps=chain.to) as mock_to:
        chain.fit_from_dataloader(dataloader, lambda x: x, device=mock_device)

        # Assert
        assert any(call.args[0] == mock_device for call in mock_to.call_args_list)
