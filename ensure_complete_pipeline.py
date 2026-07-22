"""
ensure_complete_pipeline.py
===========================
Make sure the new-FEM campaign finishes end-to-end:

  1) retry failed generate-inp
  2) solve-parallel (resume; skip existing good ODBs)
  3) extract + build dataset/train.npz
  4) train multi-output GP
  5) generate model_updating/data/dataset.npz from GP
  6) train all 5 generative methods
  7) infer_obs.py --method all  (same obs CSV for all methods)

Usage:
    python ensure_complete_pipeline.py
    python ensure_complete_pipeline.py --from-stage solve
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = [
    "retry-gen",
    "solve",
    "extract",
    "build",
    "train-gp",
    "gen-dataset",
    "train-gen",
    "infer-obs",
]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("\n>>", " ".join(str(c) for c in cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    rc = subprocess.call(cmd, cwd=str(cwd or ROOT), env=env)
    if rc != 0:
        raise SystemExit(f"FAILED ({rc}): {' '.join(str(c) for c in cmd)}")


def run_dir(rid: int) -> Path:
    return ROOT / "samples" / "runs" / f"run_{rid:04d}"


def failed_generate_ids() -> list[int]:
    ids = []
    runs = ROOT / "samples" / "runs"
    if not runs.exists():
        return ids
    for d in sorted(runs.glob("run_*")):
        rid = int(d.name.split("_")[1])
        inp = d / f"{d.name}.inp"
        if inp.exists() and inp.stat().st_size > 0:
            continue
        # missing inp -> need regenerate
        ids.append(rid)
    return ids


def mark_for_regen(rid: int) -> None:
    """Clear stub result so generate_one will re-run CAE."""
    d = run_dir(rid)
    jf = d / f"result_{rid:04d}.json"
    if jf.exists():
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        data["status"] = "retry"
        jf.write_text(json.dumps(data, indent=2), encoding="utf-8")
    # remove empty/partial inp if any
    inp = d / f"run_{rid:04d}.inp"
    if inp.exists() and inp.stat().st_size == 0:
        inp.unlink()


def stage_retry_gen(workers: int) -> None:
    ids = failed_generate_ids()
    print(f"Missing INP for {len(ids)} runs: {ids}")
    if not ids:
        return
    for rid in ids:
        mark_for_regen(rid)
    # Re-run full generate-inp; successful INPs are skipped by driver
    run([
        sys.executable, "-u", str(ROOT / "driver.py"),
        "--config", str(ROOT / "config_dataset.yaml"),
        "--n", "200", "--seed", "42",
        "--max-workers", str(workers),
        "generate-inp",
    ])
    still = failed_generate_ids()
    if still:
        print(f"WARNING: still missing INP after retry: {still}")
        print("Will retry once more...")
        for rid in still:
            mark_for_regen(rid)
            # remove cae lock if any
            for lck in run_dir(rid).glob("*.lck"):
                try:
                    lck.unlink()
                except OSError:
                    pass
        run([
            sys.executable, str(ROOT / "driver.py"),
            "--config", str(ROOT / "config_dataset.yaml"),
            "--n", "200", "--seed", "42",
            "--max-workers", "1",
            "generate-inp",
        ])
        still = failed_generate_ids()
        if still:
            raise SystemExit(f"generate-inp still failing for: {still}")


def clear_license_failed_solves() -> None:
    """If solve.log shows license error and no odb, leave as-is (resume retries).
    If a tiny/corrupt odb exists from a failed job, remove it so solve retries.
    """
    runs = ROOT / "samples" / "runs"
    for d in runs.glob("run_*"):
        rid = d.name
        odb = d / f"{rid}.odb"
        slog = d / "solve.log"
        if not slog.exists():
            continue
        txt = slog.read_text(encoding="utf-8", errors="replace")
        bad_lic = "License for standard" in txt and "not available" in txt
        bad_mem = "memory allocation request failed" in txt
        if not (bad_lic or bad_mem):
            continue
        # remove incomplete odb so solve_one will retry
        if odb.exists():
            sta = d / f"{rid}.sta"
            ok = False
            if sta.exists():
                st = sta.read_text(encoding="utf-8", errors="replace").upper()
                ok = "COMPLETED SUCCESSFULLY" in st or (
                    "COMPLETED" in st and "ABORTED" not in st)
            if not ok:
                print(f"Removing incomplete ODB for retry: {odb.name}")
                try:
                    odb.unlink()
                except OSError as e:
                    print(f"  could not remove: {e}")


def stage_solve(workers: int) -> None:
    clear_license_failed_solves()
    run([
        sys.executable, "-u", str(ROOT / "driver.py"),
        "--config", str(ROOT / "config_dataset.yaml"),
        "--n", "200", "--seed", "42",
        "--max-workers", str(workers),
        "solve-parallel",
    ])


def stage_extract(workers: int) -> None:
    run([
        sys.executable, "-u", str(ROOT / "driver.py"),
        "--config", str(ROOT / "config_dataset.yaml"),
        "--n", "200", "--seed", "42",
        "--max-workers", str(workers),
        "extract",
    ])


def stage_build() -> None:
    run([
        sys.executable, str(ROOT / "build_dataset.py"),
        "--config", str(ROOT / "config_dataset.yaml"),
    ])
    npz = ROOT / "dataset" / "train.npz"
    if not npz.exists():
        raise SystemExit("dataset/train.npz missing after build")
    data = __import__("numpy").load(npz, allow_pickle=True)
    n = len(data["run_id"])
    print(f"Built train.npz with N_ok={n}")
    if n < 150:
        print(f"WARNING: only {n}/200 successful; continuing but GP may be weaker")


def stage_train_gp() -> None:
    run([sys.executable, str(ROOT / "surrogate" / "train_gp.py")])


def stage_gen_dataset(n_gen: int) -> None:
    out = ROOT / "model_updating" / "data" / "dataset.npz"
    run([
        sys.executable, str(ROOT / "model_updating" / "generate_dataset.py"),
        "--n", str(n_gen),
        "--gp-dir", str(ROOT / "surrogate"),
        "--seed", "42",
        "--sobol",
        "--out", str(out),
    ], cwd=ROOT / "model_updating")


def stage_train_gen() -> None:
    run([
        sys.executable, str(ROOT / "model_updating" / "train.py"),
        "--method", "all",
        "--data", "data/dataset.npz",
    ], cwd=ROOT / "model_updating")


def stage_infer_obs(n_post: int) -> None:
    run([
        sys.executable, str(ROOT / "model_updating" / "infer_obs.py"),
        "--method", "all",
        "--n", str(n_post),
        "--obs-csv", "data/obs_natural_frequencies_10.csv",
        "--input-csv", "data/obs_input.csv",
        "--out", "results",
    ], cwd=ROOT / "model_updating")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-stage", choices=STAGES, default="retry-gen")
    ap.add_argument("--max-workers", type=int, default=1)
    ap.add_argument("--n-gen", type=int, default=20000)
    ap.add_argument("--n-post", type=int, default=5000,
                    help="posterior samples per observation per method")
    args = ap.parse_args()

    start = STAGES.index(args.from_stage)
    t0 = time.time()

    if start <= STAGES.index("retry-gen"):
        print("=== RETRY failed generate-inp ===")
        stage_retry_gen(args.max_workers)

    if start <= STAGES.index("solve"):
        print("=== SOLVE (resume, cpus from config) ===")
        stage_solve(args.max_workers)

    if start <= STAGES.index("extract"):
        print("=== EXTRACT ===")
        stage_extract(args.max_workers)

    if start <= STAGES.index("build"):
        print("=== BUILD train.npz ===")
        stage_build()

    if start <= STAGES.index("train-gp"):
        print("=== TRAIN GP ===")
        stage_train_gp()

    if start <= STAGES.index("gen-dataset"):
        print("=== GENERATE generative dataset from GP ===")
        stage_gen_dataset(args.n_gen)

    if start <= STAGES.index("train-gen"):
        print("=== TRAIN 5 generative methods ===")
        stage_train_gen()

    if start <= STAGES.index("infer-obs"):
        print("=== POSTERIOR on same obs for all 5 methods ===")
        stage_infer_obs(args.n_post)

    print(f"\nALL DONE in {(time.time() - t0) / 3600:.2f} h")
    print("Results: model_updating/results/")
    print("Checkpoints: model_updating/checkpoints/")


if __name__ == "__main__":
    main()
