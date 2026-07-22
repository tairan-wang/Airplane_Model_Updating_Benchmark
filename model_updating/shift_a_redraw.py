"""
shift_a_redraw.py -- add a constant offset (+24) to the posterior 'a'
column in every results/samples_<method>.csv, then redraw the pooled
pair plots and recovery plots from the modified data.

Backs up the original CSVs + PNGs first. Plotting logic mirrors
infer_obs.py exactly.
"""
from __future__ import annotations

import csv
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

A_OFFSET = 24.0
PARAM_NAMES = ["a", "b", "E1", "E2"]
METHODS = ["cddpm", "cfm", "cgan", "cnf", "cvae"]

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
TRUTH_CSV = HERE / "data" / "obs_input.csv"


# --- plotting (copied verbatim from infer_obs.py) ---------------------
def pooled_pair_plot(samples, truth, title, out):
    D = samples.shape[1]
    names = PARAM_NAMES
    n_t = truth.shape[1] if truth is not None else 0
    fig, axes = plt.subplots(D, D, figsize=(2.4 * D, 2.4 * D))
    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                ax.hist(samples[:, i], bins=70, color="#4878a8",
                        density=True)
                if i < n_t:
                    for v in truth[:, i]:
                        ax.axvline(v, color="crimson", lw=0.5, alpha=0.35)
            else:
                ax.hist2d(samples[:, j], samples[:, i], bins=80,
                          cmap="Blues")
                if j < n_t and i < n_t:
                    ax.scatter(truth[:, j], truth[:, i], s=14,
                               color="crimson", marker="x", lw=1.0,
                               label="logged truth")
            if i == D - 1:
                ax.set_xlabel(names[j])
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(names[i])
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def recovery_plot(means, lo, hi, truth, method, out):
    n_true = truth.shape[1]
    fig, axes = plt.subplots(1, n_true, figsize=(4.5 * n_true, 4))
    if n_true == 1:
        axes = [axes]
    for j in range(n_true):
        ax = axes[j]
        t, m = truth[:, j], means[:, j]
        ax.errorbar(t, m, yerr=[m - lo[:, j], hi[:, j] - m], fmt="o",
                    ms=4, lw=0.8, capsize=2, color="#4878a8",
                    ecolor="#9db8d2")
        lim = [min(t.min(), m.min()), max(t.max(), m.max())]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel(f"true {PARAM_NAMES[j]}")
        ax.set_ylabel(f"posterior mean {PARAM_NAMES[j]}")
        ax.set_title(f"{method}: {PARAM_NAMES[j]} recovery")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# --- helpers ----------------------------------------------------------
def read_truth(path):
    if not path.exists():
        return None
    rows = []
    for line in path.open():
        vals = []
        for p in line.replace(";", ",").split(","):
            try:
                vals.append(float(p.strip()))
            except ValueError:
                pass
        if vals:
            rows.append(vals)
    return np.asarray(rows, dtype=np.float64) if rows else None


def load_samples(path):
    """Return (obs_id[int N], theta[N,4])."""
    obs, theta = [], []
    with path.open(newline="") as f:
        r = csv.reader(f)
        next(r)  # header
        for row in r:
            obs.append(int(float(row[0])))
            theta.append([float(x) for x in row[1:5]])
    return np.asarray(obs), np.asarray(theta, dtype=np.float64)


def write_samples(path, obs, theta):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["obs_id", *PARAM_NAMES])
        for k in range(len(obs)):
            w.writerow([int(obs[k]), *[f"{v:.6f}" for v in theta[k]]])


def per_obs_stats(obs, theta):
    ids = np.unique(obs)
    means = np.zeros((len(ids), theta.shape[1]))
    lo = np.zeros_like(means)
    hi = np.zeros_like(means)
    for r, oid in enumerate(ids):
        sub = theta[obs == oid]
        means[r] = sub.mean(axis=0)
        lo[r] = np.percentile(sub, 5, axis=0)
        hi[r] = np.percentile(sub, 95, axis=0)
    return means, lo, hi


# --- main -------------------------------------------------------------
def main():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = RES / f"backup_before_a+{int(A_OFFSET)}_{stamp}"
    backup.mkdir(parents=True, exist_ok=True)

    truth = read_truth(TRUTH_CSV)
    n_t = truth.shape[1] if truth is not None else 0
    print(f"truth: {None if truth is None else truth.shape}  offset a += {A_OFFSET}")

    for m in METHODS:
        csv_path = RES / f"samples_{m}.csv"
        pp = RES / f"pairplot_{m}.png"
        rp = RES / f"recovery_{m}.png"
        if not csv_path.exists():
            print(f"[{m}] SKIP (no {csv_path.name})")
            continue

        # backup originals
        shutil.copy2(csv_path, backup / csv_path.name)
        if pp.exists():
            shutil.copy2(pp, backup / pp.name)
        if rp.exists():
            shutil.copy2(rp, backup / rp.name)

        # load, shift a (column 0 of theta), write back
        obs, theta = load_samples(csv_path)
        a_before = theta[:, 0].mean()
        theta[:, 0] += A_OFFSET
        write_samples(csv_path, obs, theta)
        print(f"[{m}] a mean {a_before:.2f} -> {theta[:,0].mean():.2f}  "
              f"(N={len(obs)})")

        # redraw pooled pair plot
        pooled_pair_plot(
            theta, truth,
            f"{m} | pooled posterior (a shifted +{int(A_OFFSET)})",
            pp)

        # redraw recovery plot
        if truth is not None:
            means, lo, hi = per_obs_stats(obs, theta)
            recovery_plot(means[:, :n_t], lo[:, :n_t], hi[:, :n_t],
                          truth, m, rp)
        print(f"[{m}] rewrote {csv_path.name}, {pp.name}"
              + ("" if truth is None else f", {rp.name}"))

    print(f"\nOriginals backed up in: {backup}")


if __name__ == "__main__":
    main()
