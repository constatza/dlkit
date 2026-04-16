"""Parameter partitioning strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from dlkit.common.errors import ParameterPartitionError

from .inventory import IParameterInventory, ParameterDescriptor
from .selectors import IParameterSelector


class IParameterPartitioner(ABC):
    """Abstract interface for partitioning parameters into groups.

    A partitioner takes an inventory of parameters and a sequence of selectors,
    and produces non-overlapping groups where each group contains the parameters
    that match the corresponding selector.
    """

    @abstractmethod
    def partition(
        self,
        inventory: IParameterInventory,
        selectors: Sequence[IParameterSelector],
    ) -> tuple[tuple[ParameterDescriptor, ...], ...]:
        """Partition inventory parameters into groups by selectors.

        Args:
            inventory: The parameter inventory to partition.
            selectors: Sequence of selectors in desired output order.

        Returns:
            Tuple of parameter descriptor tuples, one per selector in the same order.
            Each inner tuple contains the descriptors matching that selector.

        Raises:
            ParameterPartitionError: If the same parameter appears in multiple groups
                (overlapping selectors).
        """
        ...


class ParameterPartitioner(IParameterPartitioner):
    """Concrete partitioner that enforces non-overlapping groups.

    Filters the parameter inventory through each selector and ensures
    no parameter appears in more than one group.
    """

    def partition(
        self,
        inventory: IParameterInventory,
        selectors: Sequence[IParameterSelector],
    ) -> tuple[tuple[ParameterDescriptor, ...], ...]:
        """Partition parameters into non-overlapping groups.

        Args:
            inventory: The parameter inventory to partition.
            selectors: Sequence of selectors to apply.

        Returns:
            Tuple of parameter descriptor tuples, one per selector.

        Raises:
            ParameterPartitionError: If a parameter matches multiple selectors.
        """
        all_parameters = inventory.list_parameters()

        # Partition: apply each selector
        partitions: list[tuple[ParameterDescriptor, ...]] = []
        assigned_param_ids: set[int] = set()

        for selector in selectors:
            # Filter parameters through this selector
            matched = tuple(desc for desc in all_parameters if selector.is_satisfied_by(desc))

            # Check for overlaps with previously assigned parameters
            param_ids_in_partition = {id(desc.parameter) for desc in matched}
            overlap = assigned_param_ids & param_ids_in_partition

            if overlap:
                # Build overlap context
                overlap_names = [desc.name for desc in matched if id(desc.parameter) in overlap]
                raise ParameterPartitionError(
                    message="Parameter partitioning produced overlapping groups",
                    context={"overlap": overlap_names},
                )

            # Add to assigned set
            assigned_param_ids.update(param_ids_in_partition)
            partitions.append(matched)

        return tuple(partitions)
