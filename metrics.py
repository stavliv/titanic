import torch

def binary_accuracy(y_hat: torch.Tensor, y: torch.Tensor, threshold=0.5) -> float:
    '''
    Parameters
    ----------
    y_hat : torch.Tensor
        1d prdeictions tensor
    y : torch.Tensor
        1d labels tensor
    threshold : float
        values larger or equal to threshold are consider 1., others 0.
        
    Returns
    -------
    float
        the number of correct predictions
    '''
    return (torch.where(y_hat >= threshold, 1., 0.) == y).sum().item()