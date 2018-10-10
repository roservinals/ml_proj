import numpy as np

def compute_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)


def compute_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))


def compute_loss(y, tx, w):
    """Calculate the loss.
    """
    e = y - tx.dot(w)
    return compute_mse(e)

def compute_ls(e):
    return 1/2*np.mean(e**2)