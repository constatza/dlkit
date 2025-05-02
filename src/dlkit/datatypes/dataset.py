from torch.utils.data import Dataset
from attrs import define, field


@define
class SplitDataset:
    raw: Dataset = field()
    train: Dataset | None = field(default=None)
    validation: Dataset | None = field(default=None)
    test: Dataset | None = field(default=None)
    predict: Dataset | None = field(default=None, alias="transformed")
