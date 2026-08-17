"""
make_paper_results.py -- run the whole paper re-analysis in order, writing only
under results/paper_1000/. Reuses the existing GP + 5 checkpoints; no retraining;
does not touch the 24 mm FE offset logic.

    python make_paper_results.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

STAGES = [
    ("I+VI posterior samples + timing", "paper_posterior.py"),
    ("II  Bhattacharyya a/b", "paper_bdist_params.py"),
    ("III-A pairplots (hist+kde)", "paper_pairplots.py"),
    ("III-B a/b comparison figures", "paper_posterior_comparison.py"),
    ("IV  surrogate performance (6 modes)", "paper_surrogate_perf.py"),
    ("V-A in-domain validation (f1-f5)", "paper_validation_indomain.py"),
    ("V-B extrapolation (held-out sixth)", "paper_validation_extrap.py"),
    ("VII final metrics table", "paper_table.py"),
]


def main():
    t_all = time.time()
    for label, script in STAGES:
        print(f"\n===== {label}  ({script}) =====", flush=True)
        t0 = time.time()
        rc = subprocess.call([sys.executable, "-u", str(HERE / script)],
                             cwd=str(HERE))
        if rc != 0:
            raise SystemExit(f"FAILED: {script} (rc={rc})")
        print(f"  ({time.time() - t0:.1f}s)")
    print(f"\nAll paper results in results/paper_1000/  "
          f"({(time.time() - t_all) / 60:.1f} min)")


if __name__ == "__main__":
    main()
