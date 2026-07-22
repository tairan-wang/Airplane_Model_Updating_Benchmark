"""
validation_hist_all_modes.py -- distributional validation across all modes.

For every (surrogate mode -> observed mode) pair, overlay the histogram +
fitted (KDE) PDF of the pooled surrogate-predicted frequency (posterior-
predictive, 'a' mapped to the sketch frame) against the observed frequency.

Mode pairing (see validate_surrogate_f7.py for the ~142 Hz FE-doublet story):
    surrogate f1..f5  <->  observed f1..f5   (used in inference: consistency)
    surrogate f7      <->  observed f6       (HELD OUT: true validation)
    surrogate f6      = extra FE doublet member, no distinct obs mode (skipped)

Output: one grid PNG (rows = mode pairs, cols = methods) + a bias/RMSE table.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

import config as C
from validate_surrogate_f7 import (
    load_surrogate, read_obs_freq, load_posterior, METHODS, RES,
)

# (surrogate_mode_1based, observed_mode_1based, held_out?)
PAIRS = [
    (1, 1, False), (2, 2, False), (3, 3, False),
    (4, 4, False), (5, 5, False),
    (7, 6, True),          # matched across the FE doublet -> held-out
]

PRED_COLOR = "#4878a8"
OBS_COLOR = "#c44e52"
RNG = np.random.default_rng(0)


def kde_of(x, cap=20000):
    if len(x) > cap:
        x = x[RNG.choice(len(x), cap, replace=False)]
    return gaussian_kde(x)


def main():
    model, scaler, _ = load_surrogate()

    # predict all 7 surrogate modes once per method
    pred = {}
    for m in METHODS:
        _, th = load_posterior(m)
        th = th.copy()
        th[:, 0] -= C.A_OFFSET
        pred[m] = model.predict(scaler.transform(th))    # [N, 7]

    nR, nC = len(PAIRS), len(METHODS)
    fig, axes = plt.subplots(nR, nC, figsize=(3.2 * nC, 2.4 * nR),
                             squeeze=False)

    print(" mode-pair        method   pred_mean  obs_mean   bias[Hz]")
    for r, (sm, om, held) in enumerate(PAIRS):
        obs = read_obs_freq(f"f{om}_Hz")
        obs_kde = gaussian_kde(obs)
        for c, m in enumerate(METHODS):
            ax = axes[r][c]
            p = pred[m][:, sm - 1]
            lo, hi = min(p.min(), obs.min()), max(p.max(), obs.max())
            xs = np.linspace(lo, hi, 300)

            ax.hist(p, bins=50, density=True, color=PRED_COLOR, alpha=0.30)
            ax.plot(xs, kde_of(p)(xs), color=PRED_COLOR, lw=1.5)
            ax.hist(obs, bins=12, density=True, color=OBS_COLOR, alpha=0.30)
            ax.plot(xs, obs_kde(xs), color=OBS_COLOR, lw=1.5)
            ax.axvline(p.mean(), color=PRED_COLOR, ls="--", lw=0.8)
            ax.axvline(obs.mean(), color=OBS_COLOR, ls="--", lw=0.8)
            ax.tick_params(labelsize=6)
            ax.set_yticks([])

            if r == 0:
                ax.set_title(m, fontsize=10)
            if c == 0:
                tag = " (held-out)" if held else ""
                ax.set_ylabel(f"sur $f_{sm}$\nvs obs $f_{om}${tag}",
                              fontsize=8)
            if r == nR - 1:
                ax.set_xlabel("freq [Hz]", fontsize=7)

            bias = p.mean() - obs.mean()
            print(f"  sur f{sm} - obs f{om}   {m:6s}   {p.mean():8.1f}  "
                  f"{obs.mean():8.1f}  {bias:+6.2f}")

    # legend (proxy handles) on the top-right axis
    from matplotlib.patches import Patch
    axes[0][-1].legend(
        handles=[Patch(color=PRED_COLOR, alpha=0.6, label="surrogate (pred)"),
                 Patch(color=OBS_COLOR, alpha=0.6, label="observed")],
        fontsize=6, loc="upper right")

    fig.suptitle("Distributional validation across modes: surrogate "
                 "posterior-predictive (blue) vs observed (red)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out = RES / "validation_hist_all_modes.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
