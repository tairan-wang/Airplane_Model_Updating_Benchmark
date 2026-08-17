"""
paper_surrogate_perf.py -- Section IV. Surrogate performance figure for the six
modes reported in the paper.

The GP is NOT retrained. We reuse surrogate/test_predictions.csv (true vs
predicted for the original outputs f1..f7) and plot parity for the six paper
modes = original f1,f2,f3,f4,f5,f7 (original f6 hidden), relabelled Mode 1..6.
The original->paper index map lives in paper_common.MODE_SURR_IDX and in the
metadata written here. Output: paper_surrogate_performance_6modes.{png,pdf}.
"""
from __future__ import annotations

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

import paper_common as P


def load_test_predictions():
    rows = list(csv.DictReader((P.SUR / "test_predictions.csv").open()))
    n = 1
    while f"f{n}_true" in rows[0]:
        n += 1
    n -= 1
    yt = np.array([[float(r[f"f{i}_true"]) for i in range(1, n + 1)]
                   for r in rows])
    yp = np.array([[float(r[f"f{i}_pred"]) for i in range(1, n + 1)]
                   for r in rows])
    return yt, yp


def main():
    out = P.ensure_out()
    yt, yp = load_test_predictions()

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    r2_used = {}
    for k, si in enumerate(P.MODE_SURR_IDX):        # [0,1,2,3,4,6]
        ax = axes[k]
        t, p = yt[:, si], yp[:, si]
        r2 = float(r2_score(t, p))
        r2_used[P.MODE_LABELS[k]] = r2
        ax.scatter(t, p, s=22, color="#4878a8", alpha=0.8, edgecolors="none")
        lim = [min(t.min(), p.min()), max(t.max(), p.max())]
        pad = 0.03 * (lim[1] - lim[0] + 1e-9)
        lim = [lim[0] - pad, lim[1] + pad]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("%s  ($R^2$=%.5f)" % (P.MODE_LABELS[k], r2), fontsize=11)
        ax.set_xlabel("FE frequency [Hz]", fontsize=9)
        ax.set_ylabel("GP prediction [Hz]", fontsize=9)
        ax.tick_params(labelsize=8)
    fig.suptitle("GP surrogate performance (six reported modes)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"paper_surrogate_performance_6modes.{ext}", dpi=150)
    plt.close(fig)

    (out / "surrogate_mode_map.json").write_text(json.dumps({
        "note": "paper Mode k -> original surrogate output index (0-based); "
                "original f6 (index 5) is hidden.",
        "mode_map": {P.MODE_LABELS[k]: {"surrogate_index": P.MODE_SURR_IDX[k],
                                        "surrogate_f": P.MODE_SURR_IDX[k] + 1,
                                        "r2": r2_used[P.MODE_LABELS[k]]}
                     for k in range(len(P.MODE_LABELS))},
    }, indent=2))
    print("-> paper_surrogate_performance_6modes.{png,pdf}  |  R2:",
          {k: round(v, 5) for k, v in r2_used.items()})


if __name__ == "__main__":
    main()
