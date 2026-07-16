"""
make_training_data.py
=====================
Convenience wrapper: generate N LHS FE samples via ../driver.py then build npz.

Default N=200. This can take many hours/days (Abaqus). Prefer running
overnight with enough licenses.

Usage:
    python surrogate/make_training_data.py
    python surrogate/make_training_data.py --n 200 --max-workers 4
    python surrogate/make_training_data.py --sample-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRIVER = ROOT / "driver.py"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--sample-only", action="store_true",
                    help="only write LHS CSV, do not call Abaqus")
    ap.add_argument("--stages", nargs="+",
                    default=["sample", "generate-inp", "solve-parallel",
                             "extract", "build"],
                    help="driver stages to run in order")
    args = ap.parse_args()

    py = sys.executable
    common = [py, str(DRIVER), "--n", str(args.n), "--seed", str(args.seed),
              "--max-workers", str(args.max_workers)]

    if args.sample_only:
        stages = ["sample"]
    else:
        stages = args.stages

    for stage in stages:
        cmd = common + [stage]
        print(">>", " ".join(cmd), flush=True)
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            raise SystemExit(f"Stage {stage} failed with code {rc}")

    print("Done. Next: python surrogate/train_gp.py")


if __name__ == "__main__":
    main()
