import torch

def binary_accuracy(y_hat, y):
    return (torch.where(y_hat >= 0.5, 1., 0.) == y).sum().item()