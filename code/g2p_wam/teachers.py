"""Tensor contracts for externally computed VGGT and SpatialTrackerV2 outputs."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FeatureTargets:
    """Spatially/temporally aligned tokens [..., D] and optional weights [...]."""

    tokens: torch.Tensor
    weights: torch.Tensor | None = None

    def detached(self):
        if self.tokens.ndim < 2 or not torch.isfinite(self.tokens).all():
            raise ValueError("Expected finite teacher tokens with a feature axis")
        tokens = self.tokens.detach().float()
        weights = torch.ones_like(tokens[..., 0]) if self.weights is None else self.weights.detach().float()
        if weights.shape != tokens.shape[:-1]:
            raise ValueError("Weights must match the teacher token axes")
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Token weights must be finite and nonnegative")
        return tokens, weights.to(tokens.device)


@dataclass(frozen=True)
class TrackTargets:
    """XYZ [B,N,T,3] in one coordinate frame; visibility/confidence [B,N,T]."""

    xyz: torch.Tensor
    visibility: torch.Tensor
    confidence: torch.Tensor | None = None

    def increments(self, *, confidence_filter=False):
        if self.xyz.ndim != 4 or self.xyz.shape[-1] != 3 or self.xyz.shape[2] < 2:
            raise ValueError("Tracks must have shape [B,N,T,3] with T >= 2")
        if self.visibility.shape != self.xyz.shape[:-1]:
            raise ValueError("Visibility must match [B,N,T]")
        if not torch.isfinite(self.xyz).all() or not torch.isfinite(self.visibility).all():
            raise ValueError("Tracks and visibility must be finite")
        displacement = self.xyz.detach().float().diff(dim=2)
        visible = self.visibility.detach() > 0
        weights = (visible[:, :, :-1] & visible[:, :, 1:]).to(displacement)
        if confidence_filter:
            if self.confidence is None or self.confidence.shape != self.visibility.shape:
                raise ValueError("Confidence filtering requires [B,N,T] confidence")
            confidence = self.confidence.detach().to(displacement)
            if not torch.isfinite(confidence).all():
                raise ValueError("Track confidence must be finite")
            endpoint_confidence = torch.minimum(confidence[:, :, :-1], confidence[:, :, 1:])
            weights = weights * (endpoint_confidence >= endpoint_confidence.median())
        return displacement, weights
