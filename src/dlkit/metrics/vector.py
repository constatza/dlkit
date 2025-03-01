import torch.nn.functional as F
import torch.linalg as LA
import torch


def cosine_loss(predictions, targets):
    return 1 - F.cosine_similarity(predictions, targets, dim=-1)


def vector_norm(x, dim=-1, ord=2):
    return LA.vector_norm(x, dim=dim, ord=ord)


def sum_squares(x, dim=-1):
    return torch.sum(x**2, dim=dim)


def mean_squares(x):
    return torch.mean(x**2)


def sum_abs(x, dim=-1):
    return torch.sum(torch.abs(x), dim=dim)


def mean_abs(x, dim=-1):
    return torch.mean(torch.abs(x), dim=dim)


def rms_loss(predictions, targets):
    """Root Mean Square Error"""
    return vector_norm(predictions - targets)


def rms_over_rms_loss(predictions, targets, eps=1e-8):
    """Normalized Root Mean Square Error"""
    return rms_loss(predictions, targets) / (vector_norm(targets) + eps)
