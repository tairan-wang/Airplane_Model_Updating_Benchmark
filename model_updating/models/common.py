"""
models/common.py -- building blocks shared by all five conditional
generative models.

Every method uses the SAME ConditionEmbedder (y -> context vector h) and
implements the interface:

    loss(theta, y)      -> scalar training loss (dict for multi-loss models)
    sample(y, n)        -> n samples of theta per condition row  [n, B, D]

theta and y are ALWAYS standardised inside the framework; physical units
are restored in sample.py via the stored Standardisers.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def mlp(sizes, act=nn.SiLU, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    if out_act is not None:
        layers.append(out_act())
    return nn.Sequential(*layers)


class ConditionEmbedder(nn.Module):
    """Shared observation network: y in R^Y_DIM -> h in R^embed_dim."""

    def __init__(self, y_dim: int, hidden, embed_dim: int):
        super().__init__()
        self.net = mlp([y_dim, *hidden, embed_dim])
        self.embed_dim = embed_dim

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding followed by a small MLP (DDPM / FM)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = mlp([dim, dim * 2, dim])

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] in [0, 1] (FM) or integer steps scaled to [0, 1] (DDPM)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        ang = t.float().unsqueeze(-1) * freqs.unsqueeze(0) * 1000.0
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.proj(emb)


class ConditionalGenerativeModel(nn.Module):
    """Base class: owns the shared condition embedder."""

    def __init__(self, theta_dim: int, y_dim: int, common_cfg: dict):
        super().__init__()
        self.theta_dim = theta_dim
        self.y_dim = y_dim
        self.cond = ConditionEmbedder(
            y_dim, common_cfg["cond_hidden"], common_cfg["cond_embed_dim"])
        self.h_dim = common_cfg["cond_embed_dim"]

    def loss(self, theta: torch.Tensor, y: torch.Tensor):
        raise NotImplementedError

    @torch.no_grad()
    def sample(self, y: torch.Tensor, n: int) -> torch.Tensor:
        """Return samples with shape [n, B, theta_dim] for y of shape
        [B, y_dim]."""
        raise NotImplementedError

    def _expand_condition(self, y: torch.Tensor, n: int):
        """Embed y and tile it n times -> h of shape [n*B, h_dim]."""
        h = self.cond(y)                        # [B, H]
        B = h.shape[0]
        h = h.unsqueeze(0).expand(n, B, -1).reshape(n * B, -1)
        return h, B
