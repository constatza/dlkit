from pathlib import Path
from typing import TypeVar, List, Tuple, Annotated, Callable, Dict, Iterable

from numpy import ndarray
from pydantic import FilePath, DirectoryPath, BeforeValidator, validate_call
from torch import Tensor
from torch import Tensor
from numpy import ndarray

T = TypeVar("T")
type MaybeListLike[T] = List[T] | Tuple[T] | T
type ArrayOrTensor = Tensor | ndarray


@validate_call
def ensure_tuple(x: MaybeListLike[T], remove_nones=True) -> Tuple[T]:
    if not isinstance(x, Iterable):
        x = (x,)
    else:
        x = tuple(x)
    if remove_nones:
        x = tuple(i for i in x if i is not None)
    return x


@validate_call
def ensure_dir_exists(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


type TupleLike[T] = Annotated[Tuple[T] | List[T], BeforeValidator(ensure_tuple)]
type CreateIfNotExistsDir = Annotated[DirectoryPath, BeforeValidator(ensure_dir_exists)]
