"""models/cgan.py -- conditional GAN.

Generator G(z, h) -> theta; discriminator D(theta, h) -> logit.
Non-saturating BCE losses with decaying instance noise on real/fake
theta, which noticeably stabilises training on low-dimensional data.
Note: GANs give no likelihood and are the least statistically grounded
of the five methods -- included as a baseline for the comparison.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import ConditionalGenerativeModel, mlp


class CGAN(ConditionalGenerativeModel):
    is_adversarial = True

    def __init__(self, theta_dim, y_dim, common_cfg, cfg):
        super().__init__(theta_dim, y_dim, common_cfg)
        z = cfg["z_dim"]
        self.z_dim = z
        self.instance_noise = cfg["instance_noise"]
        self.noise_scale = cfg["instance_noise"]   # decayed by the trainer
        self.gen = mlp([z + self.h_dim, *cfg["gen_hidden"], theta_dim])
        self.disc = mlp([theta_dim + self.h_dim, *cfg["disc_hidden"], 1])
        self.bce = nn.BCEWithLogitsLoss()

    # generator / discriminator parameter groups for the two optimisers
    def gen_parameters(self):
        return list(self.gen.parameters()) + list(self.cond.parameters())

    def disc_parameters(self):
        return list(self.disc.parameters())

    def _noisy(self, x):
        if self.noise_scale > 0:
            return x + self.noise_scale * torch.randn_like(x)
        return x

    def disc_loss(self, theta, y):
        h = self.cond(y).detach()
        z = torch.randn(theta.shape[0], self.z_dim, device=theta.device)
        fake = self.gen(torch.cat([z, h], dim=-1)).detach()
        logit_real = self.disc(torch.cat([self._noisy(theta), h], dim=-1))
        logit_fake = self.disc(torch.cat([self._noisy(fake), h], dim=-1))
        return (self.bce(logit_real, torch.ones_like(logit_real))
                + self.bce(logit_fake, torch.zeros_like(logit_fake)))

    def gen_loss(self, theta, y):
        h = self.cond(y)
        z = torch.randn(theta.shape[0], self.z_dim, device=theta.device)
        fake = self.gen(torch.cat([z, h], dim=-1))
        logit = self.disc(torch.cat([self._noisy(fake), h], dim=-1))
        return self.bce(logit, torch.ones_like(logit))

    def loss(self, theta, y):          # used only for validation monitoring
        return {"loss": self.gen_loss(theta, y).detach()}

    @torch.no_grad()
    def sample(self, y, n):
        h, B = self._expand_condition(y, n)
        z = torch.randn(n * B, self.z_dim, device=h.device)
        x = self.gen(torch.cat([z, h], dim=-1))
        return x.reshape(n, B, self.theta_dim)
