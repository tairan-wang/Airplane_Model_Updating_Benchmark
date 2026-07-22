"""
regen_recovery.py -- rebuild recovery_<method>.png (new histogram+KDE style)
from the existing results/samples_<method>.csv, without re-running inference.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from infer_obs import recovery_plot, read_truth

METHODS = ["cddpm", "cfm", "cgan", "cnf", "cvae"]
RES = Path("results")
TRUTH = Path("data/obs_input.csv")


def load_pooled(method):
    th = []
    with (RES / f"samples_{method}.csv").open(newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            th.append([float(x) for x in row[1:5]])   # a, b, E1, E2
    return np.asarray(th, dtype=np.float64)


def main():
    truth = read_truth(TRUTH)
    n_t = truth.shape[1]
    for m in METHODS:
        pooled = load_pooled(m)
        out = RES / f"recovery_{m}.png"
        recovery_plot(pooled[:, :n_t], truth, m, out)
        print(f"[{m}] pooled a mean={pooled[:,0].mean():.1f} "
              f"b mean={pooled[:,1].mean():.1f}  -> {out.name}")
    print(f"truth: a mean={truth[:,0].mean():.1f} b mean={truth[:,1].mean():.1f}")


if __name__ == "__main__":
    main()
