import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    
    """
    scores: np.ndarray with shape (..., T, T)
    mask_value: float used to mask future positions (e.g., -1e9)
    Return: masked scores (same shape, dtype=float)
    """
    # Write code here
    scores = np.asarray(scores)

    if scores.ndim < 2:
        raise ValueError("scores must have at least 2 dimensions")

    if scores.shape[-2] != scores.shape[-1]:
        raise ValueError("Last two dimensions must have shape (T, T)")

    masked_scores = scores.astype(float, copy=True)

    T = scores.shape[-1]

    # True above the main diagonal, where j > i
    future_mask = np.triu(np.ones((T, T), dtype=bool), k=1)

    masked_scores[..., future_mask] = mask_value

    return masked_scores
    