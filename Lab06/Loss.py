import numpy as np

def compute_CCE_loss(AL, Y):
    # AL shape: (batch_size, n_classes)
    # Y shape: (batch_size, n_classes)
    assert AL.shape == Y.shape, f"Shape mismatch: AL shape {AL.shape}, Y shape {Y.shape}"
    assert len(AL.shape) == 2, f"AL shape should be (batch_size, n_classes), got {AL.shape}"
    eps = 1e-5
    loss = -np.sum(Y * np.log(AL + eps))
    return loss
    
    
def compute_MSE_loss(AL, Y):
    loss = np.sum((AL - Y) ** 2)
    return loss