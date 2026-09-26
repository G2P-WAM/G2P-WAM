"""Training-only readouts for static features and dynamic 3D displacement."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .teachers import FeatureTargets, TrackTargets


def weighted_average(values, weights):
    if values.shape != weights.shape:
        raise ValueError("Values and weights must have matching token axes")
    # Accumulate in float32 even when the caller uses reduced-precision tensors.
    values, weights = values.float(), weights.float()
    # An empty mask contributes zero, with a valid zero student gradient.
    mass = weights.sum()
    return (values * weights).sum() / torch.where(mass > 0, mass, torch.ones_like(mass))


class StaticReadout(nn.Module):
    """One student-layer readout; instantiate separately for additional layers."""

    def __init__(self, student_dim, teacher_dim, hidden_dim):
        super().__init__()
        self.direction = nn.Sequential(
            nn.LayerNorm(student_dim), nn.Linear(student_dim, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, teacher_dim),
        )
        self.scale = nn.Sequential(
            nn.LayerNorm(teacher_dim), nn.Linear(teacher_dim, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, teacher_dim),
        )

    def forward(self, student, target: FeatureTargets, *, scale_weight=1.0):
        if scale_weight < 0:
            raise ValueError("Scale weight must be nonnegative")
        teacher, weights = target.detached()
        projected = self.direction(student.float())
        if projected.shape != teacher.shape:
            raise ValueError("Student and teacher grids must be aligned before this readout")
        teacher, weights = teacher.to(projected), weights.to(projected)
        unit = F.normalize(projected, dim=-1)
        cosine_error = 1 - (unit * F.normalize(teacher, dim=-1)).sum(-1)
        scale_error = (self.scale(unit) - teacher).square().mean(-1)
        angular = weighted_average(cosine_error, weights)
        magnitude = weighted_average(scale_error, weights)
        return {"loss": angular + scale_weight * magnitude,
                "angular": angular, "scale": magnitude}


class DynamicReadout(nn.Module):
    """UV/time queries attend to video states and predict [B,N,S,3] increments."""

    def __init__(self, student_dim, width=64, heads=4, layers=2, bands=4, max_steps=64):
        super().__init__()
        if min(width, heads, layers, bands, max_steps) <= 0 or width % heads:
            raise ValueError("Invalid attention or embedding dimensions")
        self.register_buffer("frequencies", math.pi * 2.0 ** torch.arange(bands))
        self.video_projection = nn.Linear(student_dim, width)
        self.query_projection = nn.Linear(4 * bands, width)
        self.time_embedding = nn.Embedding(max_steps, width)
        layer = nn.TransformerDecoderLayer(
            width, heads, dim_feedforward=4 * width, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 3))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, video_states, query_uv, step_ids):
        if video_states.ndim != 3 or query_uv.ndim != 2 or query_uv.shape[-1] != 2:
            raise ValueError("Expected video [B,L,D] and normalized queries [N,2]")
        if not torch.isfinite(query_uv).all() or ((query_uv < 0) | (query_uv > 1)).any():
            raise ValueError("Query UV coordinates must lie in [0,1]")
        if step_ids.ndim != 1 or step_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Step IDs must be a one-dimensional integer tensor")
        if step_ids.numel() == 0 or query_uv.shape[0] == 0:
            raise ValueError("At least one query and one target step are required")
        if (step_ids < 0).any() or (step_ids >= self.time_embedding.num_embeddings).any():
            raise ValueError("Target step outside the configured embedding range")
        phase = query_uv.detach().to(self.frequencies)[..., None] * self.frequencies
        encoding = torch.cat((phase.sin(), phase.cos()), dim=-1).flatten(1)
        spatial = self.query_projection(encoding)
        temporal = self.time_embedding(step_ids.to(spatial.device))
        queries = (spatial[:, None] + temporal[None]).flatten(0, 1)
        memory = self.video_projection(video_states.float())
        decoded = self.decoder(queries[None].expand(memory.shape[0], -1, -1), memory)
        return self.output(decoded).reshape(memory.shape[0], len(query_uv), len(step_ids), 3)


def dynamic_loss(predicted, targets: TrackTargets, *, confidence_filter=False, beta=0.1):
    """Masked SmoothL1 summed over XYZ and averaged over valid point-time pairs."""
    if beta <= 0:
        raise ValueError("SmoothL1 beta must be positive")
    displacement, weights = targets.increments(confidence_filter=confidence_filter)
    if predicted.shape != displacement.shape:
        raise ValueError("Predictions must match the T-1 adjacent track increments")
    displacement = displacement.to(device=predicted.device, dtype=torch.float32)
    weights = weights.to(device=predicted.device, dtype=torch.float32)
    errors = F.smooth_l1_loss(predicted.float(), displacement, reduction="none", beta=beta).sum(-1)
    return weighted_average(errors, weights)
