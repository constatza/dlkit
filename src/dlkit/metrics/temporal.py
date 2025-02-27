import torch

from dlkit.metrics.vector import rms_over_rms_loss, mean_abs


def derivative(y, shift=0, n=1, dim=-1):
    """First order derivative"""
    return torch.diff(y, dim=dim, n=n)


def normalized_derivative_loss(predictions, targets):
    dx_hat = torch.diff(predictions, dim=-1, n=1)
    dx = torch.diff(targets, dim=-1, n=1)
    return normalized_mean_square_loss(dx_hat, dx)


def derivative_loss_over_rms(predictions, targets):
    dx_hat = torch.diff(predictions, dim=-1, n=1)
    dx = torch.diff(targets, dim=-1, n=1)
    return rms_over_rms_loss(dx_hat, dx)


def normalized_mean_square_loss(predictions, targets, eps=1e-8):
    """Normalized Mean Square Error"""
    return torch.mean((predictions - targets) ** 2, dim=-1) / (
        torch.mean(targets**2, dim=-1) + eps
    )


def rms_over_rms_with_derivative(predictions, targets, weight=1.0):
    derivative_loss = weight * derivative_loss_over_rms(predictions, targets)
    rms = rms_over_rms_loss(predictions, targets)
    return torch.mean(derivative_loss + rms)


def nmse_with_derivative(predictions, targets, weight=1.0):
    derivative_loss = weight * normalized_derivative_loss(predictions, targets)
    nms_loss = normalized_mean_square_loss(predictions, targets)
    return torch.mean(derivative_loss + nms_loss)


def nmse(predictions, targets):
    return torch.mean(normalized_mean_square_loss(predictions, targets))


def rmse(predictions, targets):
    return torch.mean(rms_over_rms_loss(predictions, targets))


def mas_loss(predictions, targets):
    naive_forcast_error = mean_abs(torch.diff(targets, dim=-1), dim=-1)
    return mean_abs(predictions - targets, dim=-1) / naive_forcast_error


def mase(predictions, targets):
    """Implements Mean Absolute Scaled Error (MASE).
    MASE normalizes over the naive previous value prediction.
    x_{t+1} = x_{t}
    """
    return torch.mean(mas_loss(predictions, targets))


def mas_derivative_loss(predictions, targets, n=1):
    derivative_loss = mean_abs(derivative(predictions - targets, n=n))
    naive_derivative_loss = mean_abs(derivative(targets, n=n + 1))
    return derivative_loss / naive_derivative_loss


def mase_with_derivative(predictions, targets, weight=1):
    return torch.mean(
        weight * mas_derivative_loss(predictions, targets, n=1)
        + mas_loss(predictions, targets)
    )
