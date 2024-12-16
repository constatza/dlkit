from typing import Callable

import torch
from pydantic import validate_call


class Transformation:

    def __init__(self):
        pass

    def fit(self, train):
        raise NotImplementedError("This method needs to be overridden in subclasses.")

    def fit_transform(self, train):
        self.fit(train)
        return self.transform(train)

    def transform(self, data):
        raise NotImplementedError("This method needs to be overridden in subclasses.")

    def inverse_transform(self, data):
        raise NotImplementedError("This method needs to be overridden in subclasses.")


class Map(Transformation):
    """Wrapper for a function that maps the input raw to a new space.
    No training is required for this transformation."""

    def __init__(self, func: Callable, inverse_func: Callable = None):
        super().__init__()
        self.transform: Callable = func
        self.inverse_transform: Callable = inverse_func

    def fit(self, train):
        pass


class StandardScaler(Transformation):

    def __init__(self, dim=0, keepdims=True):
        super().__init__()
        self.means = None
        self.stds = None
        self.dim = dim
        self.keepdims = keepdims

    def fit(self, train):
        self.means = train.mean(dim=self.dim, keepdims=self.keepdims)
        self.stds = train.std(dim=self.dim, keepdims=self.keepdims)
        self.stds[self.stds == 0] = 1

    def transform(self, data):
        if self.means is None or self.stds is None:
            raise RuntimeError("Scaler has not been fitted.")
        return (data - self.means) / self.stds

    @validate_call(config={"arbitrary_types_allowed": True})
    def inverse_transform(self, data: torch.Tensor):
        if self.means is None or self.stds is None:
            raise RuntimeError("Scaler has not been fitted.")
        return data * self.stds + self.means


class MinMaxScaler(Transformation):

    @validate_call
    def __init__(self, dim: tuple | int, keepdims: bool = True):
        super().__init__()
        self.mins = None
        self.maxs = None
        self.dim = dim
        self.keepdims = keepdims

    def fit(self, train):
        self.mins = torch.amin(train, dim=self.dim, keepdim=self.keepdims)
        self.maxs = torch.amax(train, dim=self.dim, keepdim=self.keepdims)
        min_eq_max = torch.where(torch.eq(self.mins, self.maxs))
        self.maxs[min_eq_max] = 1

    @validate_call(config={"arbitrary_types_allowed": True})
    def transform(self, data: torch.Tensor):
        if self.mins is None or self.maxs is None:
            raise RuntimeError("Scaler has not been fitted.")
        return (data - self.mins) / (self.maxs - self.mins)

    @validate_call(config={"arbitrary_types_allowed": True})
    def inverse_transform(self, data: torch.Tensor):
        if self.mins is None or self.maxs is None:
            raise RuntimeError("Scaler has not been fitted.")
        return data * (self.maxs - self.mins) + self.mins


class NumpyToTensor(Map):
    def __init__(self, precision=torch.float32):
        super().__init__(
            lambda x: torch.from_numpy(x).to(precision),
            inverse_func=lambda x: x.numpy(),
        )
