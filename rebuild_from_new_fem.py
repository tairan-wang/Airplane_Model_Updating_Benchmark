"""
rebuild_from_new_fem.py
=======================
Full rebuild with the updated CAE in abaqus_model/:

  1) N=200 FE LHS campaign  -> dataset/train.npz
  2) multi-output GP        -> surrogate/*.joblib
  3) GP-driven gen. dataset -> model_updating/data/dataset.npz
  4) train 5 methods        -> model_updating/checkpoints/*.pt

Usage:
    python rebuild_from_new_fem.py
    python rebuild_from_new_fem.py --n 200 --max-workers 4
    python rebuild_from_new_fem.py --from-stage train-gp   # skip FE if done
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = [
    "clean",
    "fe",
    "train-gp",
    "gen-dataset",
    "train-gen",
]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("\n>>", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    rc = subprocess.call(cmd, cwd=str(cwd or ROOT), env=env)
    if rc != 0:
        raise SystemExit(f"Command failed ({rc}): {' '.join(cmd)}")


def clean_old_outputs() -> None:
    """Remove previous FE / surrogate / generative artifacts (keep scripts & obs)."""
    targets = [
        ROOT / "samples" / "runs",
        ROOT / "dataset" / "train.npz",
        ROOT / "dataset" / "train.csv",
        ROOT / "dataset" / "manifest.json",
        ROOT / "samples" / "manifest_generate_inp.json",
        ROOT / "samples" / "manifest_solve.json",
        ROOT / "samples" / "manifest_extract.json",
        ROOT / "samples" / "active_samples.txt",
        ROOT / "surrogate" / "multioutput_gp.joblib",
        ROOT / "surrogate" / "input_scaler.joblib",
        ROOT / "surrogate" / "metrics.json",
        ROOT / "surrogate" / "r2_scores.json",
        ROOT / "surrogate" / "test_predictions.csv",
        ROOT / "surrogate" / "parity_r2.png",
        ROOT / "surrogate" / "r2_by_mode.png",
        ROOT / "surrogate" / "make_data_200.log",
        ROOT / "model_updating" / "data" / "dataset.npz",
        ROOT / "model_updating" / "checkpoints",
    ]
    for p in targets:
        if not p.exists():
            continue
        if p.is_dir():
            print(f"Removing dir  {p}")
            shutil.rmtree(p, ignore_errors=True)
        else:
            print(f"Removing file {p}")
            p.unlink(missing_ok=True)


def stage_fe(n: int, seed: int, workers: int) -> None:
    run([
        sys.executable,
        str(ROOT / "surrogate" / "make_training_data.py"),
        "--n", str(n),
        "--seed", str(seed),
        "--max-workers", str(workers),
    ])
    npz = ROOT / "dataset" / "train.npz"
    if not npz.exists():
        raise SystemExit(f"Expected {npz} after FE campaign")


def stage_train_gp() -> None:
    run([sys.executable, str(ROOT / "surrogate" / "train_gp.py")])
    if not (ROOT / "surrogate" / "multioutput_gp.joblib").exists():
        raise SystemExit("GP model missing after train_gp")


def stage_gen_dataset(n_gen: int, seed: int) -> None:
    out = ROOT / "model_updating" / "data" / "dataset.npz"
    run([
        sys.executable,
        str(ROOT / "model_updating" / "generate_dataset.py"),
        "--n", str(n_gen),
        "--gp-dir", str(ROOT / "surrogate"),
        "--seed", str(seed),
        "--sobol",
        "--out", str(out),
    ], cwd=ROOT / "model_updating")


def stage_train_gen() -> None:
    run([
        sys.executable,
        str(ROOT / "model_updating" / "train.py"),
        "--method", "all",
        "--data", "data/dataset.npz",
    ], cwd=ROOT / "model_updating")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="FE LHS samples")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--n-gen", type=int, default=20000,
                    help="synthetic pairs from GP for generative training")
    ap.add_argument("--from-stage", choices=STAGES, default="clean",
                    help="start from this stage (skip earlier)")
    ap.add_argument("--skip-clean", action="store_true")
    args = ap.parse_args()

    start = STAGES.index(args.from_stage)
    t0 = time.time()

    if start <= STAGES.index("clean") and not args.skip_clean:
        print("=== CLEAN old outputs ===")
        clean_old_outputs()

    if start <= STAGES.index("fe"):
        print("=== FE campaign (new fem_model.cae) ===")
        stage_fe(args.n, args.seed, args.max_workers)

    if start <= STAGES.index("train-gp"):
        print("=== Train multi-output GP ===")
        stage_train_gp()

    if start <= STAGES.index("gen-dataset"):
        print("=== Generate generative-model dataset from GP ===")
        stage_gen_dataset(args.n_gen, args.seed)

    if start <= STAGES.index("train-gen"):
        print("=== Train 5 generative methods ===")
        stage_train_gen()

    print(f"\nAll done in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
