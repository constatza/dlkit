from typing import TypeVar

from attrs import define, field
from pydantic import Field
from torch.utils.data import Dataset

from .basic import BasicTypeSettings

Dataset_T = TypeVar('Dataset_T', bound=Dataset, covariant=True)


class Shape(BasicTypeSettings):
	"""Shape of a tensor or array.""" ''

	features: tuple[int, ...] | None = Field(default=None)
	targets: tuple[int, ...] | None = Field(default=None)


@define
class SplitDatasetOfType[Dataset_T]:
	raw: Dataset_T = field(default=None)
	train: Dataset_T | None = field(default=None)
	validation: Dataset_T | None = field(default=None)
	test: Dataset_T | None = field(default=None)
	predict: Dataset_T | None = field(default=None, alias='transformed')


class SplitIndices(BasicTypeSettings):
	train: tuple[int, ...]
	validation: tuple[int, ...]
	test: tuple[int, ...]
