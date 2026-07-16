"""models/cddpm.py -- conditional denoising diffusion probabilistic model.

Standard DDPM (Ho et al. 2020) with a linear beta schedule, epsilon
prediction, and ancestral sampling; the noise network is conditioned on
the shared context h and a sinusoidal time embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import ConditionalGenerativeModel, TimeEmbedding, mlp


class CDDPM(ConditionalGenerativeModel):
    def __init__(self, theta_dim, y_dim, common_cfg, cfg):
        super().__init__(theta_dim, y_dim, common_cfg)
        self.T = cfg["T"]
        betas = torch.linspace(cfg["beta_start"], cfg["beta_end"], self.T)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("abar", abar)

        te = cfg["time_embed_dim"]
        self.t_embed = TimeEmbedding(te)
        self.eps_net = mlp([theta_dim + self.h_dim + te,
                            *cfg["net_hidden"], theta_dim])

    def _eps(self, x_t, t_frac, h):
        return self.eps_net(torch.cat(
            [x_t, h, self.t_embed(t_frac)], dim=-1))

    def loss(self, theta, y):
        B = theta.shape[0]
        h = self.cond(y)
        t = torch.randint(0, self.T, (B,), device=theta.device)
        a = self.abar[t].unsqueeze(-1)
        eps = torch.randn_like(theta)
        x_t = torch.sqrt(a) * theta + torch.sqrt(1.0 - a) * eps
        eps_hat = self._eps(x_t, t.float() / self.T, h)
        return {"loss": ((eps - eps_hat) ** 2).sum(dim=-1).mean()}

    @torch.no_grad()
    def sample(self, y, n):
        h, B = self._expand_condition(y, n)
        x = torch.randn(n * B, self.theta_dim, device=h.device)
        for i in reversed(range(self.T)):
            t_frac = torch.full((n * B,), i / self.T, device=h.device)
            eps_hat = self._eps(x, t_frac, h)
            alpha, abar, beta = self.alphas[i], self.abar[i], self.betas[i]
            x = (x - beta / torch.sqrt(1.0 - abar) * eps_hat) \
                / torch.sqrt(alpha)
            if i > 0:
                x = x + torch.sqrt(beta) * torch.randn_like(x)
        return x.reshape(n, B, self.theta_dim)
