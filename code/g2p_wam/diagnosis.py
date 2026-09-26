"""Descriptive diagnostics on externally supplied scores and executed outcomes."""
import math

import torch


def confidence_score(point_confidence):
    """4RC confidence [B,T,H,W] -> one mean score per video, not a probability."""
    if point_confidence.ndim != 4 or any(n == 0 for n in point_confidence.shape):
        raise ValueError("Expected a nonempty confidence map [B,T,H,W]")
    if not torch.isfinite(point_confidence).all():
        raise ValueError("Confidence maps must be finite")
    return point_confidence.detach().float().flatten(1).mean(1)


def outcome_auc(scores, outcomes):
    """P(score_success > score_failure) plus half ties; missing classes -> NaN."""
    s = torch.as_tensor(scores, device="cpu", dtype=torch.float64).detach()
    y = torch.as_tensor(outcomes).detach().to(device="cpu")
    if s.ndim != 1 or s.shape != y.shape:
        raise ValueError("Scores and outcomes must be equal-length vectors")
    if not torch.isfinite(s).all() or not ((y == 0) | (y == 1)).all():
        raise ValueError("Expected finite scores and binary executed outcomes")
    positives, negatives = s[y == 1], s[y == 0].sort().values
    if positives.numel() == 0 or negatives.numel() == 0:
        return math.nan
    below = torch.searchsorted(negatives, positives, right=False)
    at_most = torch.searchsorted(negatives, positives, right=True)
    return ((below.double() + at_most.double()) / 2).mean().item() / negatives.numel()


def score_motion_correlation(scores, motion_proxy):
    """Pearson association; the supplied motion proxy defines the interpretation."""
    x = torch.as_tensor(scores, device="cpu", dtype=torch.float64).detach()
    y = torch.as_tensor(motion_proxy, device="cpu", dtype=torch.float64).detach()
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError("Scores and motion must be equal-length vectors")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Inputs must be finite")
    if x.numel() < 2:
        return math.nan
    x, y = x - x.mean(), y - y.mean()
    denominator = x.norm() * y.norm()
    return (x.dot(y) / denominator).item() if denominator > 0 else math.nan
