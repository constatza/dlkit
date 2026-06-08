from __future__ import annotations

import unittest.mock

import torch
from tensordict import TensorDict
from torch.nn import ModuleList

from dlkit.domain.transforms.chain import TransformChain
from dlkit.domain.transforms.minmax import MinMaxScaler
from dlkit.engine.adapters.lightning.transform_pipeline import NamedBatchTransformer


def _make_batch(x: torch.Tensor) -> TensorDict:
    batch_size = [x.shape[0]]
    return TensorDict(
        {
            "features": TensorDict({"x": x}, batch_size=batch_size),
            "targets": TensorDict({}, batch_size=batch_size),
        },
        batch_size=batch_size,
    )


def test_named_batch_transformer_preserves_device_after_fit() -> None:
    # Arrange
    chain = TransformChain(ModuleList([MinMaxScaler(dim=0)]), entry_name="x")
    transformer = NamedBatchTransformer(feature_chains={"x": chain}, target_chains={})

    # Use a real device (CPU is fine for testing the logic)
    mock_device = torch.device("cpu")

    dataloader = [_make_batch(torch.tensor([[0.0, 1.0], [2.0, 3.0]]))]

    # Act
    with unittest.mock.patch.object(NamedBatchTransformer, "to", wraps=transformer.to) as mock_to:
        # Pass the explicit device
        transformer.fit(dataloader, device=mock_device)

        # Assert
        # Check that 'to' was called with the target device
        assert any(call.args[0] == mock_device for call in mock_to.call_args_list)


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
