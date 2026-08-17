"""
paper_validation_extrap.py -- Section V-B. Extrapolation validation on the
HELD-OUT SIXTH MODAL FREQUENCY.

Only f1-f5 are used for posterior conditioning. Using the existing internal
correspondence, the surrogate's high-order output (f7, index 6) is compared, as
a posterior-predictive quantity, against the experimental held-out sixth modal
frequency (observed f6_Hz). Metrics: Bhattacharyya distance (pooled
posterior-predictive density vs pooled observed density -- never a single point),
RMSE, mean bias, and 90% predictive-interval coverage.
Outputs per-method + all-methods figures and validation_extrapolation_metrics.csv.
"""
from __future__ import annotations

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

import paper_common as P

N_GRID = 4096
SI = P.HELDOUT_SURR_IDX          # surrogate f7
OBS_COL = P.HELDOUT_OBS_COL      # observed f6_Hz
PRED_C, OBS_C = "#4878a8", "#c44e52"
LABEL = "held-out sixth modal frequency"


def method_result(m, model, scaler):
    obs_id, theta = P.load_samples_csv(m)
    pcol = P.gp_predict(model, scaler, theta)[:, SI]     # [30000]
    ovals = P.read_obs_col(OBS_COL)                      # [30]
    ids = np.unique(obs_id)
    means = np.array([pcol[obs_id == i].mean() for i in ids])
    q05 = np.array([np.percentile(pcol[obs_id == i], 5) for i in ids])
    q95 = np.array([np.percentile(pcol[obs_id == i], 95) for i in ids])
    lo = min(pcol.min(), ovals.min())
    hi = max(pcol.max(), ovals.max())
    pad = 0.03 * (hi - lo + 1e-9)
    db, *_ = P.bdist_1d(pcol, ovals, lo - pad, hi + pad, N_GRID)
    return {
        "pcol": pcol, "ovals": ovals, "means": means, "q05": q05, "q95": q95,
        "DB": db,
        "RMSE": float(np.sqrt(np.mean((means - ovals) ** 2))),
        "bias": float(np.mean(means - ovals)),
        "cov": float(np.mean((q05 <= ovals) & (ovals <= q95))),
    }


def _parity(ax, r, title):
    o, mn, q05, q95 = r["ovals"], r["means"], r["q05"], r["q95"]
    ax.errorbar(o, mn, yerr=[mn - q05, q95 - mn], fmt="o", ms=4, lw=0.8,
                capsize=2, color=PRED_C, ecolor="#9db8d2")
    lim = [min(o.min(), q05.min()), max(o.max(), q95.max())]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("observed %s [Hz]" % LABEL, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)


def per_method_figure(m, r, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    _parity(axes[0], r, "%s | posterior-predicted vs observed\n"
            "$D_B$=%.3f RMSE=%.2f bias=%+.2f cov=%.2f"
            % (P.METHOD_LABEL[m], r["DB"], r["RMSE"], r["bias"], r["cov"]))
    axes[0].set_ylabel("posterior-predicted [Hz]", fontsize=8)
    # density comparison (Updated / Measured; frequency figure -> no prior)
    pcol, ovals = r["pcol"], r["ovals"]
    lo = min(pcol.min(), ovals.min())
    hi = max(pcol.max(), ovals.max())
    pad = 0.05 * (hi - lo + 1e-9)
    h = P.plot_umv(axes[1], pcol, ovals, (lo - pad, hi + pad))
    axes[1].set_xlabel("%s [Hz]" % LABEL, fontsize=8)
    axes[1].set_ylabel("PDF", fontsize=8)
    axes[1].legend(h, P.UMV_LEGEND, fontsize=7)
    axes[1].tick_params(labelsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"validation_extrapolation_{m}.{ext}", dpi=150)
    plt.close(fig)


def all_methods_figure(results, out):
    fig, axes = plt.subplots(1, len(P.METHODS), figsize=(4.0 * len(P.METHODS),
                             4.2), sharey=True)
    for ax, m in zip(axes, P.METHODS):
        r = results[m]
        _parity(ax, r, "%s\n$D_B$=%.3f RMSE=%.2f cov=%.2f"
                % (P.METHOD_LABEL[m], r["DB"], r["RMSE"], r["cov"]))
    axes[0].set_ylabel("posterior-predicted [Hz]", fontsize=9)
    fig.suptitle("Extrapolation validation: %s (observed vs posterior-predicted,"
                 " 90%% interval, y=x)" % LABEL, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"validation_extrapolation_all_methods.{ext}", dpi=150)
    plt.close(fig)


def main():
    out = P.ensure_out()
    model, scaler, _ = P.load_gp()
    results, rows = {}, []
    for m in P.METHODS:
        r = method_result(m, model, scaler)
        results[m] = r
        per_method_figure(m, r, out)
        rows.append({"method": m, "D_B": "%.6f" % r["DB"],
                     "RMSE": "%.6f" % r["RMSE"], "bias": "%.6f" % r["bias"],
                     "coverage_90": "%.4f" % r["cov"]})
        print("[%s] held-out sixth: D_B=%.4f RMSE=%.3f bias=%+.3f cov=%.2f"
              % (m, r["DB"], r["RMSE"], r["bias"], r["cov"]))
    all_methods_figure(results, out)

    path = out / "validation_extrapolation_metrics.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "D_B", "RMSE", "bias",
                                          "coverage_90"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("-> %s + per-method/all-methods figures" % path)


if __name__ == "__main__":
    main()
