"""Tests for parameter partitioning strategies."""

from __future__ import annotations

import pytest
import torch.nn as nn

from dlkit.common.errors import ParameterPartitionError
from dlkit.domain.nn.parameter_roles import ParameterRole
from dlkit.engine.training.optimization.inventory import TorchParameterInventory
from dlkit.engine.training.optimization.partitioning import ParameterPartitioner
from dlkit.engine.training.optimization.selectors import RoleSelector


@pytest.fixture
def two_selector_inventory(tiny_model: nn.Sequential) -> TorchParameterInventory:
    """Build parameter inventory from tiny_model.

    Args:
        tiny_model: Two-layer Linear model.

    Returns:
        A TorchParameterInventory wrapping the tiny model.
    """
    return TorchParameterInventory(tiny_model)


class TestParameterPartitioner:
    """Tests for ParameterPartitioner."""

    def test_partition_returns_correct_groups(
        self, two_selector_inventory: TorchParameterInventory
    ) -> None:
        """Partition with two non-overlapping role selectors returns two groups.

        Args:
            two_selector_inventory: Inventory from tiny_model.
        """
        partitioner = ParameterPartitioner()
        selectors = [
            RoleSelector(ParameterRole.UNKNOWN),
            RoleSelector(ParameterRole.HIDDEN),  # Won't match UNKNOWN-role params
        ]
        result = partitioner.partition(two_selector_inventory, selectors)

        # Should return a tuple of tuples
        assert isinstance(result, tuple)
        assert len(result) == 2

        # First partition should contain all parameters (all are UNKNOWN by default)
        # Second partition should be empty (no HIDDEN-role params)
        assert len(result[0]) > 0
        assert len(result[1]) == 0

    def test_partition_raises_on_overlap(
        self, two_selector_inventory: TorchParameterInventory
    ) -> None:
        """Partition raises ParameterPartitionError when selectors overlap.

        Args:
            two_selector_inventory: Inventory from tiny_model.
        """
        partitioner = ParameterPartitioner()
        selectors = [
            RoleSelector(ParameterRole.UNKNOWN),
            RoleSelector(ParameterRole.UNKNOWN),  # Same selector twice
        ]

        with pytest.raises(ParameterPartitionError) as exc_info:
            partitioner.partition(two_selector_inventory, selectors)

        # Check that the error message mentions overlapping groups
        assert "overlapping" in str(exc_info.value).lower()

    def test_partition_empty_selectors(
        self, two_selector_inventory: TorchParameterInventory
    ) -> None:
        """Partition with empty selector list returns empty tuple.

        Args:
            two_selector_inventory: Inventory from tiny_model.
        """
        partitioner = ParameterPartitioner()
        selectors: list = []
        result = partitioner.partition(two_selector_inventory, selectors)

        assert result == ()
