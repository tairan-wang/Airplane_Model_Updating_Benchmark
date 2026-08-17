"""
paper_bdist_params.py -- Section II. Bhattacharyya distance for the recovered
parameter distributions of a and b.

For each method: pool the 30x1000 posterior samples of a (and b), build the
reference distribution from the 30 obs_input truth values, and compute D_B with
one shared Silverman bandwidth, the same physical range, and >=4096 grid points.
No E1/E2 (no ground truth). Output: bdist_params.csv.
"""
from __future__ import annotations

import csv

import numpy as np

import paper_common as P

N_GRID = 4096


def main():
    out = P.ensure_out()
    truth = P.read_truth_ab()               # [30, 2] -> a, b (physical)
    ref = {"a": truth[:, 0], "b": truth[:, 1]}
    rng = {"a": P.A_RANGE, "b": P.B_RANGE}

    rows = []
    for m in P.METHODS:
        _, theta = P.load_samples_csv(m)     # [30000, 4]
        pooled = {"a": theta[:, 0], "b": theta[:, 1]}
        rec = {}
        for name in ("a", "b"):
            lo, hi = rng[name]
            db, *_ = P.bdist_1d(pooled[name], ref[name], lo, hi, N_GRID)
            rec[name] = db
        rows.append({"method": m, "D_B_a": rec["a"], "D_B_b": rec["b"]})
        print("[%s] D_B(a)=%.4f  D_B(b)=%.4f" % (m, rec["a"], rec["b"]))

    path = out / "bdist_params.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "D_B_a", "D_B_b"])
        w.writeheader()
        for r in rows:
            w.writerow({"method": r["method"],
                        "D_B_a": "%.6f" % r["D_B_a"],
                        "D_B_b": "%.6f" % r["D_B_b"]})
    print("-> %s" % path)


if __name__ == "__main__":
    main()
