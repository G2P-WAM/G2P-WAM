"""Dual-stream error-relative preference loss and a prediction-space anchor."""
from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class StreamPredictions:
    """Predictions on caller-conditioned winner/loser examples of equal shape."""

    winner: torch.Tensor
    loser: torch.Tensor
    reference_winner: torch.Tensor
    reference_loser: torch.Tensor
    target_winner: torch.Tensor
    target_loser: torch.Tensor
    mask: torch.Tensor | None = None


@dataclass(frozen=True)
class PreferenceLoss:
    loss: torch.Tensor
    preference: torch.Tensor
    reference_penalty: torch.Tensor
    video_logit: torch.Tensor
    action_logit: torch.Tensor


def stream_terms(stream: StreamPredictions, beta):
    """Reduce errors over the supplied microbatch before forming its logit."""
    tensors = [stream.winner, stream.loser, stream.reference_winner,
               stream.reference_loser, stream.target_winner, stream.target_loser]
    if stream.winner.numel() == 0 or any(t.shape != stream.winner.shape for t in tensors):
        raise ValueError("All stream predictions and targets must have the same nonempty shape")
    if any(t.device != stream.winner.device for t in tensors):
        raise ValueError("Stream tensors must be on the same device")
    if stream.mask is None:
        weight = None
    else:
        mask = stream.mask.detach().to(stream.winner.device, dtype=torch.float32)
        if not torch.isfinite(mask).all() or (mask < 0).any():
            raise ValueError("Mask must be finite and nonnegative")
        weight = torch.broadcast_to(mask, stream.winner.shape)

    def error(prediction, target):
        squared = (prediction.float() - target.detach().float()).square()
        if weight is None:
            return squared.mean()
        return (squared * weight).sum() / weight.sum().clamp_min(1)

    win_error = error(stream.winner, stream.target_winner)
    lose_error = error(stream.loser, stream.target_loser)
    ref_win_error = error(stream.reference_winner.detach(), stream.target_winner)
    ref_lose_error = error(stream.reference_loser.detach(), stream.target_loser)
    advantage = (ref_win_error - win_error) - (ref_lose_error - lose_error)
    logit = beta * advantage
    anchor = error(stream.winner, stream.reference_winner) + error(stream.loser, stream.reference_loser)
    return -F.logsigmoid(logit), anchor, logit


def geodpo_loss(video: StreamPredictions, action: StreamPredictions, *, beta,
                video_weight=1.0, action_weight=1.0, anchor_weight=0.0):
    """Unsmoothened preference terms plus optional reference-prediction MSE.

    Anchor is the sum over both examples and both streams, not a trajectory KL.
    Reference tensors and prediction targets are always stop-gradient inputs.
    """
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    if any(not math.isfinite(w) or w < 0 for w in (video_weight, action_weight, anchor_weight)):
        raise ValueError("Loss weights must be finite and nonnegative")
    vp, va, vz = stream_terms(video, beta)
    ap, aa, az = stream_terms(action, beta)
    preference = video_weight * vp + action_weight * ap
    anchor = va + aa
    return PreferenceLoss(preference + anchor_weight * anchor, preference, anchor, vz, az)
