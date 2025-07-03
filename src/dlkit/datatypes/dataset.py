from typing import Protocol

import torch
from attrs import define, field
from pydantic import Field, ConfigDict

from .basic import BasicTypeSettings
from torch import Size


class Shape(BasicTypeSettings):
    """Shape of a tensor or array.""" ""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    x: tuple[int, ...] | Size | None = Field(default=None, alias="features")
    y: tuple[int, ...] | Size | None = Field(default=None, alias="targets")
    edge_index: tuple[int, ...] | Size | None = Field(default=None, alias="adjacency_matrix")
    edge_attr: tuple[int, ...] | Size | None = Field(default=None, alias="edge_features")


@define
class SplitDatasetOfType[Dataset_T]:
    raw: Dataset_T = field(default=None)
    train: Dataset_T | None = field(default=None)
    validation: Dataset_T | None = field(default=None)
    test: Dataset_T | None = field(default=None)
    predict: Dataset_T | None = field(default=None)


class SplitIndices(BasicTypeSettings):
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    predict: tuple[int, ...] | None = Field(default=None)

    def __len__(self):
        return len(self.train) + len(self.validation) + len(self.test)


class DLkitDataset(Protocol):
    x: torch.Tensor
    y: torch.Tensor | None
    edge_index: torch.Tensor | None
    edge_attr: torch.Tensor | None
