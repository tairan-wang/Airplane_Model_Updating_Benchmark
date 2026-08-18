"""
paper_pairplots.py -- Section III-A. Per-method 4D pooled pairplots over
(a, b, E1, E2), in two styles:

  * histogram version  : diagonal hist+KDE, lower off-diagonal hexbin
  * KDE-only version    : diagonal KDE curves, lower off-diagonal filled contour

Physical ranges; a/b parameter bounds drawn as thin edge lines; a/b truth shown
as a few red rug marks (diagonal) / red points (a-b panel). No E1/E2 truth marks.
Outputs pairplot_hist_<m>.{png,pdf} and pairplot_kde_<m>.{png,pdf}.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

import paper_common as P

NAMES = ["a", "b", "E1", "E2"]
RANGES = [P.A_RANGE, P.B_RANGE, P.E_RANGE, P.E_RANGE]
CMAP = "Blues"
LINE = "#4878a8"          # posterior
REF_C = "#c44e52"        # reference (truth distribution)
RNG = np.random.default_rng(0)


def _kde_curve(x, xs):
    return gaussian_kde(x)(xs)


def _kde2d(x, y, xs, ys, cap=6000):
    if len(x) > cap:
        idx = RNG.choice(len(x), cap, replace=False)
        x, y = x[idx], y[idx]
    kde = gaussian_kde(np.vstack([x, y]))
    XX, YY = np.meshgrid(xs, ys)
    return XX, YY, kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)


def make_pairplot(theta, truth, method, mode, out_base):
    D = len(NAMES)
    fig, axes = plt.subplots(D, D, figsize=(2.5 * D, 2.5 * D))
    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            xr = RANGES[j]
            yr = RANGES[i]
            if i == j:
                x = theta[:, i]
                xs = np.linspace(xr[0], xr[1], 400)
                if mode == "hist":
                    ax.hist(x, bins=60, range=xr, density=True,
                            color=LINE, alpha=0.30)
                ax.plot(xs, P.kde_post(x, xs), color=LINE, lw=1.6)
                P.prior_flat(ax, xr)                      # uniform Prior PDF
                if NAMES[i] in ("a", "b") and truth is not None:
                    tv = truth[:, 0] if NAMES[i] == "a" else truth[:, 1]
                    ax.plot(tv, np.full_like(tv, 0.0), "|", color="red",
                            ms=10, mew=1.0, alpha=0.8)   # red rug (few marks)
                ax.set_xlim(xr)
                for b in xr:
                    ax.axvline(b, color="0.6", lw=0.6, ls=":")  # bounds
                ax.set_yticks([])
            else:
                x, y = theta[:, j], theta[:, i]
                xs = np.linspace(xr[0], xr[1], 120)
                ys = np.linspace(yr[0], yr[1], 120)
                if mode == "hist":
                    ax.hexbin(x, y, gridsize=45, cmap=CMAP,
                              extent=(xr[0], xr[1], yr[0], yr[1]), mincnt=1)
                else:
                    XX, YY, ZZ = _kde2d(x, y, xs, ys)
                    ax.contourf(XX, YY, ZZ, levels=10, cmap=CMAP)
                if (NAMES[j] in ("a", "b") and NAMES[i] in ("a", "b")
                        and truth is not None):
                    tj = truth[:, 0] if NAMES[j] == "a" else truth[:, 1]
                    ti = truth[:, 0] if NAMES[i] == "a" else truth[:, 1]
                    ax.scatter(tj, ti, s=14, color="red", marker="x", lw=1.0)
                ax.set_xlim(xr)
                ax.set_ylim(yr)
                for b in xr:
                    ax.axvline(b, color="0.6", lw=0.6, ls=":")
                for b in yr:
                    ax.axhline(b, color="0.6", lw=0.6, ls=":")
            if i == D - 1:
                ax.set_xlabel(NAMES[j] + (" [mm]" if NAMES[j] in ("a", "b")
                              else r" [$\times10^{11}$ Pa]"), fontsize=9)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(NAMES[i] + (" [mm]" if NAMES[i] in ("a", "b")
                              else r" [$\times10^{11}$ Pa]"), fontsize=9)
            elif j == 0:
                ax.set_ylabel("density", fontsize=9)
            ax.tick_params(labelsize=7)
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], color=LINE, lw=1.6, label="posterior"),
           Line2D([0], [0], color="red", marker="|", ls="none",
                  label="target (a, b)"),
           Line2D([0], [0], color=P.PRIOR_PDF, lw=1.2, ls="--",
                  label="Prior PDF")]
    axes[0, 0].legend(handles=leg, fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("%s | pooled posterior (%s)"
                 % (P.METHOD_LABEL[method],
                    "histogram" if mode == "hist" else "KDE"), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=150)
    plt.close(fig)


def make_pairplot_compare(theta, truth, method, out_base):
    """Posterior vs reference as distributions: for a and b, overlay the
    reference (truth) as histogram+KDE (diagonal) and 2D KDE contour (a-b
    panel) instead of individual points. E1/E2 have no truth -> posterior only.
    """
    D = len(NAMES)
    truth_col = {"a": truth[:, 0], "b": truth[:, 1]}
    fig, axes = plt.subplots(D, D, figsize=(2.5 * D, 2.5 * D))
    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            xr, yr = RANGES[j], RANGES[i]
            if i == j:
                x = theta[:, i]
                xs = np.linspace(xr[0], xr[1], 400)
                ax.hist(x, bins=60, range=xr, density=True, color=LINE,
                        alpha=0.30)
                ax.plot(xs, P.kde_post(x, xs), color=LINE, lw=1.6,
                        label="posterior")
                P.prior_flat(ax, xr)                      # uniform Prior PDF
                if NAMES[i] in ("a", "b"):
                    tv = truth_col[NAMES[i]]
                    ax.hist(tv, bins=15, range=xr, density=True, color=REF_C,
                            alpha=0.22)
                    ax.plot(xs, P.kde_tgt(tv, xs), color=REF_C, lw=1.6, ls="-")
                ax.set_xlim(xr)
                ax.set_yticks([])
            else:
                x, y = theta[:, j], theta[:, i]
                ax.hexbin(x, y, gridsize=45, cmap=CMAP,
                          extent=(xr[0], xr[1], yr[0], yr[1]), mincnt=1)
                if NAMES[j] in ("a", "b") and NAMES[i] in ("a", "b"):
                    tj, ti = truth_col[NAMES[j]], truth_col[NAMES[i]]
                    xs2 = np.linspace(xr[0], xr[1], 120)
                    ys2 = np.linspace(yr[0], yr[1], 120)
                    XX, YY, ZZ = _kde2d(tj, ti, xs2, ys2, cap=len(tj))
                    ax.contour(XX, YY, ZZ, levels=5, colors=REF_C,
                               linewidths=0.9, linestyles="--")
                ax.set_xlim(xr)
                ax.set_ylim(yr)
            if i == D - 1:
                ax.set_xlabel(NAMES[j] + (" [mm]" if NAMES[j] in ("a", "b")
                              else r" [$\times10^{11}$ Pa]"), fontsize=9)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(NAMES[i] + (" [mm]" if NAMES[i] in ("a", "b")
                              else r" [$\times10^{11}$ Pa]"), fontsize=9)
            elif j == 0:
                ax.set_ylabel("density", fontsize=9)
            ax.tick_params(labelsize=7)
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], color=LINE, lw=1.6, label="posterior"),
           Line2D([0], [0], color=REF_C, lw=1.6, label="target (a, b)"),
           Line2D([0], [0], color=P.PRIOR_PDF, lw=1.2, ls="--",
                  label="Prior PDF")]
    axes[0, 0].legend(handles=leg, fontsize=8, frameon=False)
    fig.suptitle("%s | pooled posterior vs target (histogram + KDE)"
                 % P.METHOD_LABEL[method], fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=150)
    plt.close(fig)


def main():
    out = P.ensure_out()
    truth = P.read_truth_ab()
    for m in P.METHODS:
        _, theta = P.load_samples_csv(m)
        make_pairplot(theta, truth, m, "hist", out / f"pairplot_hist_{m}")
        make_pairplot(theta, truth, m, "kde", out / f"pairplot_kde_{m}")
        make_pairplot_compare(theta, truth, m, out / f"pairplot_compare_{m}")
        print("[%s] pairplot_hist/kde/compare_%s" % (m, m))
    print("-> %s" % out)


if __name__ == "__main__":
    main()
