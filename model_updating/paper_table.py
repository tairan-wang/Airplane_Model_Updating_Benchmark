"""
paper_table.py -- Section VII. Assemble the final paper metrics table from the
already-computed artifacts. Emits paper_table_metrics.csv and .tex.

Columns (NO E1/E2 truth metrics anywhere):
  Method | Training time (s) | Posterior-gen time 30x1000 (mean+/-std s)
  | D_B(a) | D_B(b)
  | mean in-domain D_B(f1-f5) | mean in-domain RMSE(f1-f5) | mean in-domain 90% cov
  | held-out sixth D_B | held-out sixth RMSE | held-out sixth bias | held-out sixth 90% cov

Training time is READ from checkpoint stats['train_time_s']; if absent it is left
blank (never fabricated).
"""
from __future__ import annotations

import csv
import json

import torch

import paper_common as P

OUT = P.OUT


def read_csv_by_method(path):
    d = {}
    for r in csv.DictReader(path.open()):
        d[r["method"]] = r
    return d


def train_time(method):
    try:
        ck = torch.load(P.CKPT / f"{method}.pt", map_location="cpu",
                        weights_only=False)
        tt = ck.get("stats", {}).get("train_time_s")
        return float(tt) if tt is not None else None
    except Exception:
        return None


def main():
    timing = json.loads((OUT / "timing_posterior.json").read_text())["per_method"]
    bparams = read_csv_by_method(OUT / "bdist_params.csv")
    indom = read_csv_by_method(OUT / "validation_indomain_metrics.csv")
    extrap = read_csv_by_method(OUT / "validation_extrapolation_metrics.csv")

    cols = [
        ("method", "Method"),
        ("train_time_s", "Training time (s)"),
        ("post_gen_s", "Posterior-gen 30x1000 (s)"),
        ("DB_a", "D_B(a)"),
        ("DB_b", "D_B(b)"),
        ("indomain_DB", "mean in-domain D_B(f1-f5)"),
        ("indomain_RMSE", "mean in-domain RMSE(f1-f5) [Hz]"),
        ("indomain_cov", "mean in-domain 90% cov(f1-f5)"),
        ("heldout_DB", "held-out sixth D_B"),
        ("heldout_RMSE", "held-out sixth RMSE [Hz]"),
        ("heldout_bias", "held-out sixth bias [Hz]"),
        ("heldout_cov", "held-out sixth 90% cov"),
    ]

    rows = []
    for m in P.METHODS:
        tt = train_time(m)
        t = timing[m]
        rows.append({
            "method": P.METHOD_LABEL[m],
            "train_time_s": ("%.1f" % tt) if tt is not None else "",
            "post_gen_s": "%.3f +/- %.3f" % (t["mean_s"], t["std_s"]),
            "DB_a": "%.4f" % float(bparams[m]["D_B_a"]),
            "DB_b": "%.4f" % float(bparams[m]["D_B_b"]),
            "indomain_DB": "%.4f" % float(indom[m]["mean_DB"]),
            "indomain_RMSE": "%.3f" % float(indom[m]["mean_RMSE"]),
            "indomain_cov": "%.2f" % float(indom[m]["mean_cov"]),
            "heldout_DB": "%.4f" % float(extrap[m]["D_B"]),
            "heldout_RMSE": "%.3f" % float(extrap[m]["RMSE"]),
            "heldout_bias": "%+.3f" % float(extrap[m]["bias"]),
            "heldout_cov": "%.2f" % float(extrap[m]["coverage_90"]),
        })

    # CSV
    csv_path = OUT / "paper_table_metrics.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k, _ in cols])
        w.writerow({k: h for k, h in cols})       # header row = readable labels
        for r in rows:
            w.writerow(r)

    # LaTeX (booktabs)
    tex = ["\\begin{tabular}{l" + "r" * (len(cols) - 1) + "}",
           "\\toprule",
           " & ".join(h.replace("_", "\\_").replace("%", "\\%")
                     for _, h in cols) + " \\\\",
           "\\midrule"]
    for r in rows:
        tex.append(" & ".join(str(r[k]).replace("+/-", "$\\pm$")
                              for k, _ in cols) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "paper_table_metrics.tex").write_text("\n".join(tex))

    print("-> paper_table_metrics.csv + paper_table_metrics.tex")
    for r in rows:
        print("  %-6s train=%-6s gen=%-16s D_B(a)=%s D_B(b)=%s "
              "in[DB=%s RMSE=%s cov=%s] out[DB=%s RMSE=%s bias=%s cov=%s]"
              % (r["method"], r["train_time_s"], r["post_gen_s"], r["DB_a"],
                 r["DB_b"], r["indomain_DB"], r["indomain_RMSE"],
                 r["indomain_cov"], r["heldout_DB"], r["heldout_RMSE"],
                 r["heldout_bias"], r["heldout_cov"]))


if __name__ == "__main__":
    main()
