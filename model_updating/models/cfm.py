"""models/cfm.py -- conditional flow matching (rectified / CFM path).

Learns a conditional velocity field v(theta_t, t, h) along the linear
interpolant x_t = (1 - (1 - sigma_min) t) x0 + t x1 between base noise x0
and data x1 (Lipman et al. 2023). Sampling integrates the learned ODE
from t = 0 (noise) to t = 1 (posterior sample) with fixed-step Euler.

Independent (x0, x1) coupling is used here; minibatch OT coupling
(OT-CFM, Tong et al. 2024) is a drop-in extension: reorder x0 within the
batch by solving an OT assignment before computing the interpolant.
"""

from __future__ import annotations

import torch

from .common import ConditionalGenerativeModel, TimeEmbedding, mlp


class CFM(ConditionalGenerativeModel):
    def __init__(self, theta_dim, y_dim, common_cfg, cfg):
        super().__init__(theta_dim, y_dim, common_cfg)
        self.sigma_min = cfg["sigma_min"]
        self.ode_steps = cfg["ode_steps"]
        te = cfg["time_embed_dim"]
        self.t_embed = TimeEmbedding(te)
        self.v_net = mlp([theta_dim + self.h_dim + te,
                          *cfg["net_hidden"], theta_dim])

    def _v(self, x, t, h):
        return self.v_net(torch.cat([x, h, self.t_embed(t)], dim=-1))

    def loss(self, theta, y):
        B = theta.shape[0]
        h = self.cond(y)
        x1 = theta
        x0 = torch.randn_like(x1)
        t = torch.rand(B, device=theta.device)
        tt = t.unsqueeze(-1)
        x_t = (1.0 - (1.0 - self.sigma_min) * tt) * x0 + tt * x1
        u_t = x1 - (1.0 - self.sigma_min) * x0          # target velocity
        v_hat = self._v(x_t, t, h)
        return {"loss": ((v_hat - u_t) ** 2).sum(dim=-1).mean()}

    @torch.no_grad()
    def sample(self, y, n):
        h, B = self._expand_condition(y, n)
        x = torch.randn(n * B, self.theta_dim, device=h.device)
        dt = 1.0 / self.ode_steps
        for k in range(self.ode_steps):
            t = torch.full((n * B,), k * dt, device=h.device)
            x = x + dt * self._v(x, t, h)               # Euler step
        return x.reshape(n, B, self.theta_dim)
