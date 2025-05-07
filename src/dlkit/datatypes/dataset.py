from pydantic import Field
from torch.utils.data import Dataset
from attrs import define, field

from .basic import BasicTypeSettings


class Shape(BasicTypeSettings):
	"""Shape of a tensor or array.""" ''

	features: tuple[int, ...] | None = Field(default=None)
	targets: tuple[int, ...] | None = Field(default=None)


@define
class SplitDataset:
	raw: Dataset = field()
	train: Dataset | None = field(default=None)
	validation: Dataset | None = field(default=None)
	test: Dataset | None = field(default=None)
	predict: Dataset | None = field(default=None, alias='transformed')


class SplitIndices(BasicTypeSettings):
	train: tuple[int, ...]
	validation: tuple[int, ...]
	test: tuple[int, ...]
