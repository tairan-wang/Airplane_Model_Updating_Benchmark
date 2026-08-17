"""
paper_validation_indomain.py -- Section V-A. In-domain validation on the first
five modal frequencies (f1-f5), which were used for posterior conditioning.

Posterior samples are pushed back through the GP surrogate (physical a -> sketch
inside gp_predict) to obtain posterior-predictive f1-f5. For each method/mode:
  * Bhattacharyya distance between the POOLED posterior-predictive frequency
    density (30,000) and the POOLED observed frequency density (30 values -> KDE)
    -- a single observed frequency is never treated as a density;
  * per-observation RMSE, mean bias, and 90% predictive-interval coverage.
Outputs per-method + all-methods figures and validation_indomain_metrics.csv.
"""
from __future__ import annotations

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import paper_common as P

N_GRID = 4096
MODES = P.INDOMAIN_SURR_IDX          # [0,1,2,3,4] -> f1..f5
MLAB = [f"f{i + 1}" for i in MODES]
PRED_C, OBS_C = "#4878a8", "#c44e52"


def method_metrics(m, model, scaler):
    """Return dict: per-mode D_B/RMSE/bias/coverage + pooled arrays for plotting."""
    obs_id, theta = P.load_samples_csv(m)
    pred = P.gp_predict(model, scaler, theta)        # [30000, 7]
    ids = np.unique(obs_id)
    res = {"pred": {}, "obs": {}, "DB": {}, "RMSE": {}, "bias": {}, "cov": {}}
    for mi in MODES:
        pcol = pred[:, mi]
        ovals = P.read_obs_col(f"f{mi + 1}_Hz")      # [30]
        lo = min(pcol.min(), ovals.min())
        hi = max(pcol.max(), ovals.max())
        pad = 0.03 * (hi - lo + 1e-9)
        db, *_ = P.bdist_1d(pcol, ovals, lo - pad, hi + pad, N_GRID)
        means = np.array([pcol[obs_id == i].mean() for i in ids])
        q05 = np.array([np.percentile(pcol[obs_id == i], 5) for i in ids])
        q95 = np.array([np.percentile(pcol[obs_id == i], 95) for i in ids])
        res["pred"][mi] = pcol
        res["obs"][mi] = ovals
        res["DB"][mi] = db
        res["RMSE"][mi] = float(np.sqrt(np.mean((means - ovals) ** 2)))
        res["bias"][mi] = float(np.mean(means - ovals))
        res["cov"][mi] = float(np.mean((q05 <= ovals) & (ovals <= q95)))
    return res


def per_method_figure(m, res, out):
    """Updated/Measured overlay per mode (f1-f5) + shared legend panel.
    Frequency figures carry NO prior."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6))
    axes = axes.ravel()
    handles = None
    for k, mi in enumerate(MODES):
        ax = axes[k]
        pcol, ovals = res["pred"][mi], res["obs"][mi]
        lo = min(pcol.min(), ovals.min())
        hi = max(pcol.max(), ovals.max())
        pad = 0.05 * (hi - lo + 1e-9)
        h = P.plot_umv(ax, pcol, ovals, (lo - pad, hi + pad))
        handles = handles or h
        ax.set_xlabel(r"$f_%d$ [Hz]" % (mi + 1), fontsize=11)
        ax.set_ylabel("PDF", fontsize=11)
        ax.set_title("$D_B$=%.3f  RMSE=%.2f  cov=%.2f"
                     % (res["DB"][mi], res["RMSE"][mi], res["cov"][mi]),
                     fontsize=9)
        ax.tick_params(labelsize=8)
    axes[5].axis("off")
    axes[5].legend(handles, P.UMV_LEGEND, loc="center", frameon=True,
                   fontsize=11)
    fig.suptitle("%s | posterior-prediction validation (f1-f5)"
                 % P.METHOD_LABEL[m], fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"validation_indomain_{m}.{ext}", dpi=150)
    plt.close(fig)


def summary_figure(agg, out):
    labels = [P.METHOD_LABEL[m] for m in P.METHODS]
    colors = [P.METHOD_COLOR[m] for m in P.METHODS]
    x = np.arange(len(P.METHODS))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, title in [
        (axes[0], "mean_DB", "mean $D_B$ (f1-f5)"),
        (axes[1], "mean_RMSE", "mean RMSE (f1-f5) [Hz]"),
        (axes[2], "mean_cov", "mean 90% coverage (f1-f5)"),
    ]:
        vals = [agg[m][key] for m in P.METHODS]
        ax.bar(x, vals, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, fontsize=9)
        ax.set_title(title, fontsize=11)
        if key == "mean_cov":
            ax.axhline(0.9, color="k", ls="--", lw=0.8)
        for xi, v in zip(x, vals):
            ax.text(xi, v, "%.3f" % v, ha="center", va="bottom", fontsize=7)
    fig.suptitle("In-domain validation summary (f1-f5)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"validation_indomain_all_methods.{ext}", dpi=150)
    plt.close(fig)


def main():
    out = P.ensure_out()
    model, scaler, _ = P.load_gp()
    agg = {}
    rows = []
    for m in P.METHODS:
        res = method_metrics(m, model, scaler)
        per_method_figure(m, res, out)
        mean_db = float(np.mean([res["DB"][mi] for mi in MODES]))
        mean_rmse = float(np.mean([res["RMSE"][mi] for mi in MODES]))
        mean_cov = float(np.mean([res["cov"][mi] for mi in MODES]))
        agg[m] = {"mean_DB": mean_db, "mean_RMSE": mean_rmse,
                  "mean_cov": mean_cov}
        row = {"method": m}
        for mi, lab in zip(MODES, MLAB):
            row[f"DB_{lab}"] = "%.6f" % res["DB"][mi]
            row[f"RMSE_{lab}"] = "%.6f" % res["RMSE"][mi]
            row[f"bias_{lab}"] = "%.6f" % res["bias"][mi]
            row[f"cov_{lab}"] = "%.4f" % res["cov"][mi]
        row["mean_DB"] = "%.6f" % mean_db
        row["mean_RMSE"] = "%.6f" % mean_rmse
        row["mean_cov"] = "%.4f" % mean_cov
        rows.append(row)
        print("[%s] mean D_B=%.4f mean RMSE=%.3f mean cov=%.2f"
              % (m, mean_db, mean_rmse, mean_cov))
    summary_figure(agg, out)

    path = out / "validation_indomain_metrics.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("-> %s + per-method/all-methods figures" % path)


if __name__ == "__main__":
    main()
