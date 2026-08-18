"""
paper_posterior_comparison.py -- Section III-B. All-methods comparison of the
pooled posterior marginals for a and b only, against the reference distribution.

  * posterior_comparison_hist_ab_all_methods.{png,pdf} : overlaid histograms
  * posterior_comparison_kde_ab_all_methods.{png,pdf}  : KDE curves only (main)

Colour-blind-safe (Okabe-Ito). a in physical [290,310] mm, b in [20,30] mm.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

import paper_common as P

PARAMS = [("a", P.A_RANGE, "a [mm]"), ("b", P.B_RANGE, "b [mm]")]
COL = {"a": 0, "b": 1}


def _load_all():
    data = {}
    for m in P.METHODS:
        _, theta = P.load_samples_csv(m)
        data[m] = theta
    return data


def make_hist(data, truth, out_base):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (name, rng, lab) in zip(axes, PARAMS):
        bins = np.linspace(rng[0], rng[1], 45)
        for m in P.METHODS:
            ax.hist(data[m][:, COL[name]], bins=bins, density=True,
                    histtype="step", lw=1.6, color=P.METHOD_COLOR[m],
                    label=P.METHOD_LABEL[m])
        ref = truth[:, COL[name]]
        ax.hist(ref, bins=bins, density=True, histtype="stepfilled",
                color="#777777", alpha=0.12)
        ax.hist(ref, bins=bins, density=True, histtype="step",
                lw=1.8, color="#444444", ls="-", label="target")
        P.prior_flat(ax, rng, label="Prior PDF")     # shared uniform Prior PDF
        ax.set_xlim(rng)
        ax.set_xlabel(lab)
        ax.set_ylabel("probability density")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Pooled posterior vs target (histograms)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=150)
    plt.close(fig)


def make_kde(data, truth, out_base):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (name, rng, lab) in zip(axes, PARAMS):
        xs = np.linspace(rng[0], rng[1], 512)
        for m in P.METHODS:                              # smoothed posterior KDEs
            ax.plot(xs, P.kde_post(data[m][:, COL[name]], xs), lw=1.8,
                    color=P.METHOD_COLOR[m], label=P.METHOD_LABEL[m])
        ax.plot(xs, P.kde_tgt(truth[:, COL[name]], xs), lw=2.0, color="#444444",
                ls="-", label="target")
        P.prior_flat(ax, rng, label="Prior PDF")     # shared uniform Prior PDF
        ax.set_xlim(rng)
        ax.set_xlabel(lab)
        ax.set_ylabel("probability density")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Pooled posterior vs target (KDE)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=150)
    plt.close(fig)


def main():
    out = P.ensure_out()
    data = _load_all()
    truth = P.read_truth_ab()
    make_hist(data, truth, out / "posterior_comparison_hist_ab_all_methods")
    make_kde(data, truth, out / "posterior_comparison_kde_ab_all_methods")
    print("-> posterior_comparison_{hist,kde}_ab_all_methods.{png,pdf}")


if __name__ == "__main__":
    main()
