from functools import partial
from typing import Literal

import torch
import torch.nn.functional as F

from dlkit.metrics.vector import (
    mean_abs,
    mean_squares,
    rms_over_rms_loss,
    sum_squares,
    vector_norm,
)


def apply(func, dim: int | list[int] = 1, aggregator=vector_norm):
    def wrapper(*args, **kwargs):
        return aggregator(func(*args, **kwargs), dim=dim)

    return wrapper


def add_with_weight(func1, func2):
    def wrapper(*args, weight=1, **kwargs):
        return func1(*args, **kwargs) + weight * func2(*args, **kwargs)

    return wrapper


def derivative(y, n=1, dim=-1):
    """First order derivative"""
    return torch.diff(y, dim=dim, n=n)


def time_mean_abs_loss(predictions, targets):
    return mean_abs(predictions - targets, dim=2)


def time_mean_sqr_loss(predictions, targets):
    return mean_squares(predictions - targets, dim=2)


def derivative_mean_sqr_loss(predictions, targets, n=1):
    return mean_squares(derivative(predictions - targets, n=n), dim=2)


def derivative_mean_abs_loss(predictions, targets, n=1):
    return mean_abs(derivative(predictions - targets, n=n), dim=2)


def l2_squared_loss(predictions, targets, dim=1):
    return sum_squares(predictions - targets, dim=dim)


def vectorized_sum_sqr_loss(predictions, targets, dim=1):
    return sum_squares(predictions - targets, dim=dim)


def mean_error(loss_func, dim: int | list[int] = 1):

    def wrapper(predictions, targets, *args, **kwargs):
        return torch.mean(loss_func(predictions, targets, *args, **kwargs), dim=dim)

    return wrapper


def naive_forecast_scaler(loss_func, dim: int | list[int] = 1):

    def wrapper(predictions, targets, *args, **kwargs):
        return mean_error(loss_func, dim=dim)(
            targets[:, :, :-1], targets[:, :, 1:], *args, **kwargs
        )

    return wrapper


def std_scaler(loss_func, dim: int | list[int] = 1):

    def wrapper(predictions, targets, *args, **kwargs):
        return torch.std(loss_func(predictions, targets, *args, **kwargs), dim=dim)

    return wrapper


def mean_scaled_error(
    loss_func,
    scaler: callable = naive_forecast_scaler,
    dim: int | list[int] = 1,
):

    def wrapper(predictions, targets, *args, **kwargs):
        error = mean_error(loss_func, dim=dim)(predictions, targets, *args, **kwargs)
        return error / scaler(loss_func, dim=dim)(predictions, targets)

    return wrapper


mase = mean_scaled_error(time_mean_abs_loss, dim=[0, 1])
msse = mean_scaled_error(time_mean_sqr_loss, dim=[0, 1])
derivative_abs_error = mean_scaled_error(derivative_mean_abs_loss, dim=[0, 1])
derivative_sqr_error = mean_scaled_error(derivative_mean_sqr_loss, dim=[0, 1])

mase_with_derivative = add_with_weight(mase, derivative_abs_error)
msse_with_derivative = add_with_weight(msse, derivative_sqr_error)

mean_vectorized_scaled_error = mean_scaled_error(vectorized_sum_sqr_loss, dim=[0, 1])
