"""models/cvae.py -- conditional variational autoencoder.

Encoder q(z | theta, h) and decoder p(theta | z, h); trained on the ELBO
with a beta-weighted KL term. Sampling: z ~ N(0, I) -> decoder mean.
The decoder outputs a Gaussian with learned diagonal variance, so the
posterior spread is represented both by z and by the decoder noise.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import ConditionalGenerativeModel, mlp


class CVAE(ConditionalGenerativeModel):
    def __init__(self, theta_dim, y_dim, common_cfg, cfg):
        super().__init__(theta_dim, y_dim, common_cfg)
        z = cfg["z_dim"]
        self.z_dim = z
        self.beta = cfg["beta"]
        self.enc = mlp([theta_dim + self.h_dim, *cfg["enc_hidden"], 2 * z])
        self.dec = mlp([z + self.h_dim, *cfg["dec_hidden"], 2 * theta_dim])

    # ------------------------------------------------------------------
    def _decode(self, z, h):
        out = self.dec(torch.cat([z, h], dim=-1))
        mu, log_var = out.chunk(2, dim=-1)
        log_var = log_var.clamp(-8.0, 4.0)
        return mu, log_var

    def loss(self, theta, y):
        h = self.cond(y)
        mu_z, log_var_z = self.enc(
            torch.cat([theta, h], dim=-1)).chunk(2, dim=-1)
        log_var_z = log_var_z.clamp(-8.0, 4.0)
        std_z = torch.exp(0.5 * log_var_z)
        z = mu_z + std_z * torch.randn_like(std_z)

        mu_x, log_var_x = self._decode(z, h)
        # Gaussian NLL reconstruction
        rec = 0.5 * ((theta - mu_x) ** 2 / log_var_x.exp()
                     + log_var_x).sum(dim=-1).mean()
        kl = 0.5 * (mu_z ** 2 + log_var_z.exp() - 1.0
                    - log_var_z).sum(dim=-1).mean()
        return {"loss": rec + self.beta * kl, "rec": rec.detach(),
                "kl": kl.detach()}

    @torch.no_grad()
    def sample(self, y, n):
        h, B = self._expand_condition(y, n)
        z = torch.randn(n * B, self.z_dim, device=h.device)
        mu_x, log_var_x = self._decode(z, h)
        x = mu_x + torch.exp(0.5 * log_var_x) * torch.randn_like(mu_x)
        return x.reshape(n, B, self.theta_dim)
