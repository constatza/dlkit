from typing import Any

import torch
from loguru import logger
from pydantic import FilePath
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import BaseTransform, GCNNorm
from torch_geometric.utils import dense_to_sparse

from dlkit.infrastructure.io import load_array
from dlkit.infrastructure.precision.service import get_precision_service

from .base import BaseDataset, register_dataset
from .tensor_utils import ensure2d

PROCESSED_FILE_NAMES: list[str] = ["graph_data.pt"]


def build_data_list(
    *,
    x: Tensor,
    y: Tensor | None,
    edge_index: Tensor | None,
    edge_attr: Tensor | None,
) -> list[Data]:
    data_items: list[Data] = []
    for i in range(x.shape[0]):
        xi = ensure2d(x[i])
        yi = ensure2d(y[i]) if y is not None else None

        if yi is None:
            data_items.append(
                Data(
                    x=xi,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    num_nodes=int(xi.shape[0]),
                )
            )
        else:
            data_items.append(
                Data(
                    x=xi,
                    y=yi,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    num_nodes=int(xi.shape[0]),
                )
            )
    return data_items


@register_dataset
class GraphDataset(InMemoryDataset, BaseDataset[Data]):
    def __init__(
        self,
        x: FilePath,
        edge_index: FilePath,
        y: FilePath | None = None,
        **kwargs,
    ) -> None:
        self._raw_paths: dict[str, FilePath] = {"x": x, "edge_index": edge_index}
        if y is not None:
            self._raw_paths["y"] = y
        precision_service = get_precision_service()
        self._target_dtype = precision_service.get_torch_dtype()
        # Precision is automatically resolved from global precision service
        # which checks precision context (set via precision_override())
        self.x = load_array(x, dtype=self._target_dtype)
        self.y = load_array(y, dtype=self._target_dtype) if y else None
        self.adjacency = load_array(edge_index, dtype=torch.long)
        self._edge_index_shape: tuple[int, ...] | None = None
        self._edge_attr_shape: tuple[int, ...] | None = None
        super().__init__(**kwargs)
        self.load(self.processed_paths[0])
        self._apply_precision_to_cached_data()

    @property
    def raw_file_names(self) -> list[str]:
        return [file.name for file in self._raw_paths.values()]

    @property
    def raw_file_paths(self) -> dict[str, FilePath]:
        return self._raw_paths

    @property
    def processed_file_names(self) -> list[str]:
        return PROCESSED_FILE_NAMES

    @property
    def edge_index_shape(self) -> tuple[int, ...] | None:
        """Shape of the edge_index tensor after sparse conversion.

        Returns:
            Tuple representing the shape of edge_index (typically (2, num_edges)),
            or None if the dataset hasn't been processed yet.
        """
        return self._edge_index_shape

    @property
    def edge_attr_shape(self) -> tuple[int, ...] | None:
        """Shape of the edge_attr tensor after sparse conversion.

        Returns:
            Tuple representing the shape of edge_attr (typically (num_edges, edge_features)),
            or None if no edge attributes exist or dataset hasn't been processed yet.
        """
        return self._edge_attr_shape

    def process(self) -> None:
        edge_index, edge_attr = dense_to_sparse(self.adjacency)
        self._edge_index_shape = tuple(edge_index.shape)
        self._edge_attr_shape = tuple(edge_attr.shape) if edge_attr is not None else None

        data_list = build_data_list(x=self.x, y=self.y, edge_index=edge_index, edge_attr=edge_attr)
        data_list = [self._cast_graph_data_precision(d) for d in data_list]

        if self.pre_transform:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_paths[0])
        self._apply_precision_to_cached_data()

    def get(self, idx: int) -> Data:
        from typing import cast as _cast

        return self._cast_graph_data_precision(_cast(Data, super().get(idx)))

    # --------------------------------------------------------------------- #
    # Precision utilities
    # --------------------------------------------------------------------- #
    def _cast_graph_data_precision(self, data: Data) -> Data:
        def _convert(value: Tensor) -> Tensor:
            if isinstance(value, Tensor) and value.is_floating_point():
                return value.to(dtype=self._target_dtype)
            return value

        return data.apply(_convert)

    def _apply_precision_to_cached_data(self) -> None:
        data = getattr(self, "_data", None)
        if data is None:
            return
        try:
            converted = self._cast_graph_data_precision(data)
            self._data = converted
        except Exception as exc:
            logger.warning(
                "Failed to apply target dtype %s to cached graph data (%s): %s",
                self._target_dtype,
                type(data),
                exc,
            )


class ScaledGraphDataset(GraphDataset):
    def __init__(
        self,
        x: FilePath,
        edge_index: FilePath,
        y: FilePath | None = None,
        pre_transform: BaseTransform | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            x=x,
            edge_index=edge_index,
            y=y,
            pre_transform=pre_transform or GCNNorm(),
            **kwargs,
        )
