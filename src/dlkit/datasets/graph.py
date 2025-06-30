import numpy as np
import torch
from pydantic import FilePath
from torch import Tensor
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.transforms import GCNNorm
from torch_geometric.utils import dense_to_sparse
from dlkit.datatypes.dataset import Shape
from dlkit.io import load_array


PROCESSED_FILE_NAMES: list[str] = ["graph_data.pt"]


def build_data_list(
    features_np: np.ndarray,
    targets_np: np.ndarray,
    edge_index: Tensor,
    edge_attr: Tensor,
) -> list[Data]:
    data_items: list[Data] = []
    for i in range(features_np.shape[0]):
        x = torch.tensor(features_np[i], dtype=torch.float)
        y = torch.tensor(targets_np[i], dtype=torch.long)
        data_items.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return data_items


class GraphDataset(InMemoryDataset):
    def __init__(
        self,
        x: FilePath,
        edge_index: FilePath,
        y: FilePath | None = None,
        **kwargs,
    ):
        self._raw_paths = {"x": x, "edge_index": edge_index, "y": y}
        self.features = load_array(x, dtype=torch.float)
        self.targets = load_array(y, dtype=torch.long)
        self.adjacency = load_array(edge_index, dtype=torch.long)
        super().__init__(**kwargs)
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

        data_list = build_data_list(self.features, self.targets, edge_index, edge_attr)

        if self.pre_transform:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_paths[0])

    @property
    def shape(self) -> Shape:
        return Shape(
            features=self.features.shape,
            targets=self.targets.shape,
            edge_index=self.adjacency.shape,
            edge_attr=self.adjacency.shape,
        )


class ScaledGraphDataset(GraphDataset):
    def __init__(self, pre_transform=GCNNorm(), **kwargs):
        super().__init__(pre_transform=pre_transform, **kwargs)
