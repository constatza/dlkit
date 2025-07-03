import torch
from pydantic import FilePath
from torch import Tensor
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.transforms import GCNNorm
from torch_geometric.utils import dense_to_sparse
from dlkit.datatypes.dataset import Shape
from dlkit.io import load_array
from dlkit.utils.torch_utils import ensure2d
from .base import register_dataset, BaseDataset


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
        yi = ensure2d(y[i])

        data_items.append(Data(x=xi, y=yi, edge_index=edge_index, edge_attr=edge_attr))
    return data_items


@register_dataset
class GraphDataset(InMemoryDataset, BaseDataset):
    def __init__(
        self,
        x: FilePath,
        edge_index: FilePath,
        y: FilePath | None = None,
        **kwargs,
    ):
        self._raw_paths = {"x": x, "edge_index": edge_index, "y": y}
        self.x = load_array(x, dtype=torch.float)
        self.y = load_array(y, dtype=torch.float) if y else None
        self.adjacency = load_array(edge_index, dtype=torch.long)
        super().__init__(**kwargs)
        self.process()
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list[str]:
        return [file.name for file in self._raw_paths.values()]

    @property
    def raw_file_paths(self) -> dict[str, FilePath]:
        return self._raw_paths

    @property
    def processed_file_names(self) -> list[str]:
        return PROCESSED_FILE_NAMES

    def process(self) -> None:
        edge_index, edge_attr = dense_to_sparse(self.adjacency)

        data_list = build_data_list(x=self.x, y=self.y, edge_index=edge_index, edge_attr=edge_attr)

        if self.pre_transform:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_paths[0])

    @property
    def shape(self) -> Shape:
        return Shape(
            x=self.x.shape,
            y=self.y.shape,
            edge_attr=self.edge_attr.shape,
            edge_index=self.edge_index.shape,
        )


class ScaledGraphDataset(GraphDataset):
    def __init__(self, pre_transform=GCNNorm(), **kwargs):
        super().__init__(pre_transform=pre_transform, **kwargs)
