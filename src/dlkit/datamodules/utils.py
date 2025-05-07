import torch
import json

from pydantic import validate_call, FilePath, DirectoryPath
from pathlib import Path
from loguru import logger

from dlkit.datatypes.dataset import SplitIndices


def generate_split(
	n: int,
	test_size: float,
	val_size: float,
	seed: int | None = None,
) -> SplitIndices:
	"""Create immutable train/val/test index splits for a dataset of size n."""
	gen = torch.Generator()
	if seed is not None:
		gen = gen.manual_seed(seed)
	perm = torch.randperm(n, generator=gen)
	test_count = int(n * test_size)
	val_count = int(n * val_size)
	train_count = n - test_count - val_count

	return SplitIndices(
		train=tuple(perm[:train_count].tolist()),
		validation=tuple(perm[train_count : train_count + val_count].tolist()),
		test=tuple(perm[train_count + val_count :].tolist()),
	)


# --- I/O actions for split file ---
def load_split_indices(path: Path) -> SplitIndices:
	"""Load train/val/test indices from a JSON file."""
	with path.open('r') as f:
		raw = json.load(f)
	return SplitIndices(train=raw['train'], validation=raw['validation'], test=raw['test'])


def save_split_indices(
	idx_split: SplitIndices,
	path: Path,
) -> None:
	"""Save index splits to a JSON file, adding 'idx_path' metadata."""
	path.parent.mkdir(parents=True, exist_ok=True)
	data = idx_split.model_dump()
	with path.open('w') as f:
		json.dump(data, f)


@validate_call
def get_or_create_idx_split(
	n: int,
	filepath: FilePath | None = None,
	save_dir: DirectoryPath | None = None,
	test_size: float = 0.15,
	val_size: float = 0.15,
	seed: int | None = None,
) -> (SplitIndices, FilePath):
	"""Load existing split if available (validated by pydantic), otherwise generate and save a new one.

	Args:
	    filepath: Optional FilePath to an existing split JSON. If provided, must exist.
	    save_dir: DirectoryPath to save the split JSON file. If idx_split is provided, it is ignored.
	    n: Total number of samples.
	    test_size: Fraction for test set.
	    val_size: Fraction for validation set.
	    seed: Random seed for reproducibility.

	Returns:
	    A tuple of (split mapping, path to the split JSON file).
	"""
	if filepath is not None:
		idx_split = load_split_indices(filepath)
		logger.info(f'Loaded indices from {filepath}')
		return idx_split

	if save_dir is None:
		raise ValueError('Either save_dir or idx_split must be provided.')
	idx_split = generate_split(n=n, test_size=test_size, val_size=val_size, seed=seed)
	filepath = Path(save_dir) / 'idx_split.json'
	save_split_indices(idx_split, filepath)
	logger.info(f'No split file provided. Created new split at {filepath}')
	return idx_split
