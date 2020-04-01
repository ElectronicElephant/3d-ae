import torch
import torch.nn.functional as F


def dae_loss(input, target, weight=None, reduction='mean'):
    loss = F.binary_cross_entropy(input, target, reduction=reduction)
    return loss


def vae_loss(input, target, weight=None, reduction='mean', mu=None, logvar=None):
    loss1 = F.binary_cross_entropy(input, target, reduction=reduction)
    loss2 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return loss1 + loss2
