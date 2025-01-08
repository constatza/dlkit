import torch


def root_mean_square(tensor):
    return torch.sqrt(torch.mean(tensor**2))

def rmse_loss(predictions, targets):
    """Root Mean Square Error"""
    return torch.sqrt(torch.nn.functional.mse_loss(predictions, targets))

def nrmse_loss(predictions, targets):
    """Normalized Root Mean Square Error"""
    return rmse_loss(predictions, targets) / root_mean_square(targets)

# Reconstruction + KL divergence losses summed over all elements and batch
def mse_plus_kl_divergence(recon_x, x, mu, logvar, kld_weight=1e-1):
    mse = torch.nn.MSELoss()(recon_x, x)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # print((kld / mse).detach())
    return mse + kld_weight * kld
