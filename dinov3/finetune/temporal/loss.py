import torch
import torch.nn.functional as F


def masks(artists, days, ids, families, window_days=365):
    """Ignore distant same-artist pairs and all self/duplicate-work pairs."""
    same = artists[:, None] == artists[None, :]
    different_work = (ids[:, None] != ids[None, :]) & (families[:, None] != families[None, :])
    near = (days[:, None] - days[None, :]).abs() <= window_days
    return same & different_work & near, (~same) & different_work


def sigmoid_loss(features, temperature, bias, positive, negative):
    """Sum signed logistic pair losses / image count (not valid-pair count)."""
    features = F.normalize(features.float(), dim=1)
    logits = temperature * (features @ features.T) + bias
    sign = torch.where(positive, 1.0, -1.0)
    return (F.softplus(-sign * logits) * (positive | negative)).sum() / len(features)
