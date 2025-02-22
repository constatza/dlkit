import torch
import torch.nn.functional as F
import torch.linalg as LA


def rmse_loss(predictions, targets):
    """Root Mean Square Error"""
    return LA.vector_norm(predictions - targets, dim=-1, ord=2)


def nmse_loss(predictions, targets, eps=1e-8):
    """Normalized Mean Square Error"""
    return torch.mean((predictions - targets) ** 2, dim=-1) / (
        torch.var(targets, dim=-1) + eps
    )


def nrmse_loss(predictions, targets):
    """Normalized Root Mean Square Error"""
    return rmse_loss(predictions, targets) / (LA.norm(targets, dim=-1) + 1e-8)


def nmse_time_series_loss(predictions, targets):
    return torch.mean(nmse_loss(predictions, targets))


def cosine_loss(predictions, targets):
    return 1 - F.cosine_similarity(predictions, targets, dim=-1)


# Reconstruction + KL divergence losses summed over all elements and batch
def mse_plus_kl_divergence(recon_x, x, mu, logvar, kld_weight=1e-1):
    mse = torch.nn.MSELoss()(recon_x, x)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # print((kld / mse).detach())
    return mse + kld_weight * torch.kl_div(recon_x, x, reduction=1)


def sym_huber_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    delta: float = 1.0,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Computes a symmetric Huber error between predictions and targets.

    Args:
        y_pred (torch.Tensor): Predicted values (any shape).
        y_true (torch.Tensor): Ground-truth values (same shape as y_pred).
        delta (float): The threshold parameter for Huber loss transition.
        epsilon (float): Small constant for numerical stability in denominator.

    Returns:
        torch.Tensor: A scalar representing the symmetric Huber error
                      in [0, 2] if data is nonnegative, but can exceed that range otherwise.
    """
    diff = y_pred - y_true
    abs_diff = torch.abs(diff)

    # Standard Huber numerator
    quadratic = 0.5 * (diff**2)
    linear = delta * abs_diff - 0.5 * (delta**2)
    huber_per_element = torch.where(abs_diff <= delta, quadratic, linear)

    # Symmetric form
    numerator = huber_per_element
    denominator = torch.abs(y_pred) + torch.abs(y_true) + epsilon
    return torch.mean(numerator / denominator)


def sym_squared_loss(
    y_pred: torch.Tensor, y_true: torch.Tensor, epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Computes a symmetric squared error between predictions and targets.

    Args:
        y_pred (torch.Tensor): Predicted values (any shape).
        y_true (torch.Tensor): Ground-truth values (same shape as y_pred).
        epsilon (float): Small constant for numerical stability in denominator.

    Returns:
        torch.Tensor: A scalar representing the symmetric squared error in [0, 2].
    """
    diff = y_pred - y_true
    numerator = diff**2
    denominator = (y_pred**2) + (y_true**2) + epsilon
    return torch.mean(numerator / denominator)


import torch


def smape_loss(
    y_pred: torch.Tensor, y_true: torch.Tensor, epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Computes the Symmetric Mean Absolute Percentage Error (SMAPE).

    Args:
        y_pred (torch.Tensor): Predicted values (any shape).
        y_true (torch.Tensor): Ground-truth values (same shape as y_pred).
        epsilon (float): Small constant for numerical stability in denominator.

    Returns:
        torch.Tensor: A scalar representing the SMAPE in [0, 2].
                      Multiply by 100 if you want percentage form.
    """
    numerator = torch.abs(y_pred - y_true)
    denominator = torch.abs(y_pred) + torch.abs(y_true) + epsilon
    return torch.mean(numerator / denominator)
