"""
regen_recovery.py -- rebuild recovery_<method>.png (histogram+KDE style) AND
results/summary.csv from the existing results/samples_<method>.csv, without
re-running inference. Uses the same helpers as infer_obs.py so the output is
identical to a fresh `infer_obs.py --method all` run.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from infer_obs import recovery_plot, read_truth, per_obs_summary, write_summary

METHODS = ["cddpm", "cfm", "cgan", "cnf", "cvae"]
RES = Path("results")
TRUTH = Path("data/obs_input.csv")


def load_samples(method):
    """Return (obs_id[N], theta[N,4]) from samples_<method>.csv."""
    obs, th = [], []
    with (RES / f"samples_{method}.csv").open(newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            obs.append(int(float(row[0])))
            th.append([float(x) for x in row[1:5]])
    return np.asarray(obs), np.asarray(th, dtype=np.float64)


def to_th_phys(obs, theta):
    """Group pooled samples back into [n_samples, N_obs, 4]."""
    ids = np.unique(obs)
    per = [theta[obs == i] for i in ids]
    n = min(len(p) for p in per)                 # guard against ragged counts
    return np.stack([p[:n] for p in per], axis=1)   # [n, N_obs, 4]


def main():
    truth = read_truth(TRUTH)
    n_t = truth.shape[1] if truth is not None else 0
    summary_rows = []
    for m in METHODS:
        obs, theta = load_samples(m)
        recovery_plot(theta[:, :n_t], truth, m, RES / f"recovery_{m}.png")
        th_phys = to_th_phys(obs, theta)
        summary_rows.extend(per_obs_summary(m, th_phys, truth))
        print(f"[{m}] pooled a={theta[:,0].mean():.1f} b={theta[:,1].mean():.1f}"
              f"  -> recovery_{m}.png")
    write_summary(summary_rows, RES / "summary.csv", n_t)
    if truth is not None:
        print(f"truth: a={truth[:,0].mean():.1f} b={truth[:,1].mean():.1f}")


if __name__ == "__main__":
    main()
