"""models/cnf.py -- conditional normalising flow.

RealNVP-style affine coupling layers over theta (4D), with every coupling
network conditioned on the shared context h. Exact log-likelihood
training (this is the NPE-style density estimator of the comparison);
sampling by inverting the flow from base Gaussian noise.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import ConditionalGenerativeModel, mlp


class AffineCoupling(nn.Module):
    def __init__(self, dim, h_dim, hidden, mask):
        super().__init__()
        self.register_buffer("mask", mask)          # 1 = pass-through
        self.net = mlp([dim + h_dim, *hidden, 2 * dim])

    def forward(self, x, h):                         # x -> u, returns logdet
        xm = x * self.mask
        s, t = self.net(torch.cat([xm, h], dim=-1)).chunk(2, dim=-1)
        s = torch.tanh(s) * 2.0                      # clamp for stability
        u = xm + (1.0 - self.mask) * (x * torch.exp(s) + t)
        logdet = ((1.0 - self.mask) * s).sum(dim=-1)
        return u, logdet

    def inverse(self, u, h):
        um = u * self.mask
        s, t = self.net(torch.cat([um, h], dim=-1)).chunk(2, dim=-1)
        s = torch.tanh(s) * 2.0
        x = um + (1.0 - self.mask) * ((u - t) * torch.exp(-s))
        return x


class CNF(ConditionalGenerativeModel):
    def __init__(self, theta_dim, y_dim, common_cfg, cfg):
        super().__init__(theta_dim, y_dim, common_cfg)
        masks = []
        for k in range(cfg["n_coupling"]):
            m = torch.zeros(theta_dim)
            m[k % 2::2] = 1.0                        # alternate halves
            masks.append(m)
        self.layers = nn.ModuleList(
            AffineCoupling(theta_dim, self.h_dim, cfg["coupling_hidden"], m)
            for m in masks)

    def log_prob(self, theta, y):
        h = self.cond(y)
        u, logdet = theta, 0.0
        for layer in self.layers:
            u, ld = layer(u, h)
            logdet = logdet + ld
        log_base = -0.5 * (u ** 2 + torch.log(
            torch.tensor(2.0 * torch.pi, device=u.device))).sum(dim=-1)
        return log_base + logdet

    def loss(self, theta, y):
        nll = -self.log_prob(theta, y).mean()
        return {"loss": nll}

    @torch.no_grad()
    def sample(self, y, n):
        h, B = self._expand_condition(y, n)
        u = torch.randn(n * B, self.theta_dim, device=h.device)
        for layer in reversed(self.layers):
            u = layer.inverse(u, h)
        return u.reshape(n, B, self.theta_dim)
