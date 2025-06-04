from attrs import define, field
from pydantic import Field

from .basic import BasicTypeSettings


class Shape(BasicTypeSettings):
    """Shape of a tensor or array.""" ""

    features: tuple[int, ...] | None = Field(default=None)
    targets: tuple[int, ...] | None = Field(default=None)


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
