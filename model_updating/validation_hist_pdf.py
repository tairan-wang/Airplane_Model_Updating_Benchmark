"""
validation_hist_pdf.py -- distributional view of the held-out validation.

For each method, overlay:
  * histogram + fitted (KDE) PDF of the pooled surrogate-predicted f7
    (posterior-predictive over all observations, 'a' mapped to sketch frame),
  * histogram + fitted (KDE) PDF of the observed f6 (the physically matched
    mode; see validate_surrogate_f7.py for the pairing rationale).

A good validation = the two densities overlap.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

import config as C
from validate_surrogate_f7 import (
    load_surrogate, read_obs_freq, load_posterior,
    SUR_MODE, SUR_IDX, OBS_MODE, OBS_COL, METHODS, RES,
)

PRED_COLOR = "#4878a8"
OBS_COLOR = "#c44e52"


def main():
    model, scaler, _ = load_surrogate()
    obs = read_obs_freq(OBS_COL)                       # [N_obs] observed f6
    obs_kde = gaussian_kde(obs)

    fig, axes = plt.subplots(1, len(METHODS),
                             figsize=(4.0 * len(METHODS), 4.2), sharey=True)
    for ax, m in zip(axes, METHODS):
        obs_id, th = load_posterior(m)
        th = th.copy()
        th[:, 0] -= C.A_OFFSET                         # physical -> sketch
        pred = model.predict(scaler.transform(th))[:, SUR_IDX]   # pooled f7

        # common x grid spanning both distributions
        lo = min(pred.min(), obs.min())
        hi = max(pred.max(), obs.max())
        xs = np.linspace(lo, hi, 400)

        # predicted: histogram + KDE (subsample for a fast KDE)
        sub = pred if len(pred) <= 20000 else \
            pred[np.random.default_rng(0).choice(len(pred), 20000,
                                                  replace=False)]
        pred_kde = gaussian_kde(sub)
        ax.hist(pred, bins=60, density=True, color=PRED_COLOR, alpha=0.30)
        ax.plot(xs, pred_kde(xs), color=PRED_COLOR, lw=1.8,
                label=f"surrogate $f_{SUR_MODE}$ (predicted)")

        # observed: histogram + KDE (30 points)
        ax.hist(obs, bins=12, density=True, color=OBS_COLOR, alpha=0.30)
        ax.plot(xs, obs_kde(xs), color=OBS_COLOR, lw=1.8,
                label=f"observed $f_{OBS_MODE}$")

        ax.axvline(pred.mean(), color=PRED_COLOR, ls="--", lw=1)
        ax.axvline(obs.mean(), color=OBS_COLOR, ls="--", lw=1)
        ax.set_title(f"{m}\npred mean={pred.mean():.1f}  "
                     f"obs mean={obs.mean():.1f} Hz")
        ax.set_xlabel("frequency [Hz]")
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("probability density")
    axes[-1].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Distributional validation: surrogate $f_{SUR_MODE}$ "
                 f"(posterior-predictive)  vs  observed $f_{OBS_MODE}$",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = RES / f"validation_hist_surf{SUR_MODE}_vs_obsf{OBS_MODE}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
