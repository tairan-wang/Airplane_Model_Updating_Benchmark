"""
driver.py
=========
Three-stage dataset pipeline for generative-model training:

  1. sample        — 4D Latin Hypercube (uniform prior) -> samples/lhs_*.csv
  2. generate-inp  — CAE noGUI write_inp (geometry a,b + materials E1,E2)
  3. solve-parallel— local parallel `abaqus job=... input=....inp` (no CAE)
  4. extract       — `abaqus python extract_odb.py` per ODB -> result JSON
  5. build         — aggregate JSON -> dataset/train.npz

Also:
  python driver.py all          # sample + generate-inp + solve + extract + build
  python driver.py legacy-full  # old one-shot cae noGUI full solve (serial)

Config: config_dataset.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parent


# ----------------------------------------------------------------------
# Config helpers
# ----------------------------------------------------------------------
def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Resolve relative paths against package root
    for key in ("samples_dir", "runs_dir", "dataset_dir",
                "script_inp", "script_extract"):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str((ROOT / cfg[key]).resolve())
    if "master_cae" in cfg:
        cfg["master_cae"] = str(Path(cfg["master_cae"]))
    return cfg


def e_1e11_to_mpa(e_1e11: float) -> float:
    """Table 6 units (x 1e11 Pa) -> Abaqus material table [MPa]."""
    return float(e_1e11) * 1e5


def samples_csv_path(cfg: dict) -> Path:
    samples_dir = Path(cfg["samples_dir"])
    n = int(cfg["n_samples"])
    seed = int(cfg["seed"])
    return samples_dir / f"lhs_{n}_seed{seed}.csv"


def run_dir(cfg: dict, run_id: int) -> Path:
    return Path(cfg["runs_dir"]) / f"run_{run_id:04d}"


def read_samples(csv_path: Path) -> list[dict[str, float]]:
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "run_id": int(row["run_id"]),
                "a": float(row["a"]),
                "b": float(row["b"]),
                "E1_1e11Pa": float(row["E1_1e11Pa"]),
                "E2_1e11Pa": float(row["E2_1e11Pa"]),
                "E1_MPa": float(row["E1_MPa"]),
                "E2_MPa": float(row["E2_MPa"]),
            })
    return rows


def write_manifest(cfg: dict, name: str, payload: dict) -> None:
    path = Path(cfg["samples_dir"]) / f"manifest_{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Manifest -> {path}")


def run_cmd(
    cmd: list[str],
    cwd: Path,
    timeout_s: int,
    log_path: Path | None = None,
) -> int:
    """Run a command; on Windows use shell so abaqus.bat resolves."""
    use_shell = sys.platform == "win32"
    if use_shell:
        cmdline = subprocess.list2cmdline(cmd)
    else:
        cmdline = cmd
    try:
        proc = subprocess.run(
            cmdline,
            cwd=str(cwd),
            timeout=timeout_s,
            shell=use_shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if log_path is not None:
            log_path.write_text(proc.stdout or "", encoding="utf-8",
                                errors="replace")
        return int(proc.returncode)
    except subprocess.TimeoutExpired as exc:
        if log_path is not None:
            out = ""
            if exc.stdout:
                out = exc.stdout if isinstance(exc.stdout, str) else \
                    exc.stdout.decode("utf-8", errors="replace")
            log_path.write_text(out + "\nTIMEOUT\n", encoding="utf-8")
        return -9


# ----------------------------------------------------------------------
# Stage: sample (LHS)
# ----------------------------------------------------------------------
def cmd_sample(cfg: dict) -> Path:
    bounds = cfg["bounds"]
    n = int(cfg["n_samples"])
    seed = int(cfg["seed"])
    lo = np.array([bounds["a"][0], bounds["b"][0],
                   bounds["E1"][0], bounds["E2"][0]], dtype=float)
    hi = np.array([bounds["a"][1], bounds["b"][1],
                   bounds["E1"][1], bounds["E2"][1]], dtype=float)

    sampler = qmc.LatinHypercube(d=4, seed=seed)
    u = sampler.random(n=n)
    theta = qmc.scale(u, lo, hi)

    out = samples_csv_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "run_id", "a", "b",
            "E1_1e11Pa", "E2_1e11Pa", "E1_MPa", "E2_MPa",
        ])
        for i in range(n):
            a, b, e1, e2 = theta[i]
            w.writerow([
                i,
                f"{a:.8g}", f"{b:.8g}",
                f"{e1:.8g}", f"{e2:.8g}",
                f"{e_1e11_to_mpa(e1):.8g}", f"{e_1e11_to_mpa(e2):.8g}",
            ])
    # Pointer so later stages find this CSV even if config.n_samples differs
    pointer = Path(cfg["samples_dir"]) / "active_samples.txt"
    pointer.write_text(str(out.resolve()), encoding="utf-8")
    print(f"LHS samples -> {out}  (N={n}, seed={seed})")
    return out


def resolve_samples_csv(cfg: dict) -> Path:
    """Prefer active_samples.txt from the last `sample` run."""
    pointer = Path(cfg["samples_dir"]) / "active_samples.txt"
    if pointer.exists():
        p = Path(pointer.read_text(encoding="utf-8").strip())
        if p.exists():
            return p
    path = samples_csv_path(cfg)
    if path.exists():
        return path
    raise FileNotFoundError(
        f"No samples CSV. Run: python driver.py sample  "
        f"(looked for {path})")


# ----------------------------------------------------------------------
# Stage: generate-inp
# ----------------------------------------------------------------------
def _prepare_run_dir(cfg: dict, sample: dict) -> Path:
    rid = int(sample["run_id"])
    rd = run_dir(cfg, rid)
    rd.mkdir(parents=True, exist_ok=True)

    script = Path(cfg["script_inp"])
    if not script.exists():
        script = ROOT / "abaqus_run_single.py"
    shutil.copy(script, rd / "abaqus_run_single.py")

    extract = Path(cfg.get("script_extract", ROOT / "extract_odb.py"))
    if not extract.exists():
        extract = ROOT / "extract_odb.py"
    if extract.exists():
        shutil.copy(extract, rd / "extract_odb.py")

    master = Path(cfg["master_cae"])
    local_name = cfg.get("local_cae_name", "master_model.cae")
    local_cae = rd / local_name
    if not local_cae.exists():
        if not master.exists():
            raise FileNotFoundError(
                f"Master CAE not found: {master}. Update master_cae in "
                f"config_dataset.yaml")
        shutil.copy(master, local_cae)
    return rd


def generate_one(cfg: dict, sample: dict) -> dict:
    rid = int(sample["run_id"])
    rd = _prepare_run_dir(cfg, sample)
    inp_path = rd / f"run_{rid:04d}.inp"
    stub = rd / f"result_{rid:04d}.json"

    # Resume: already have INP
    if inp_path.exists() and inp_path.stat().st_size > 0:
        return {"run_id": rid, "status": "inp_ready", "skipped": True}

    cmd = [
        str(cfg.get("abaqus_cmd", "abaqus")),
        "cae",
        "noGUI=abaqus_run_single.py",
        "--",
        str(rid),
        f"{sample['a']:.6f}",
        f"{sample['b']:.6f}",
        f"{sample['E1_MPa']:.3f}",
        f"{sample['E2_MPa']:.3f}",
        "write_inp",
    ]
    rc = run_cmd(cmd, rd, int(cfg.get("timeout_gen_s", 1800)),
                 rd / "generate_inp.log")

    # Enrich stub JSON with Table-6 units if CAE wrote a result
    if stub.exists():
        try:
            data = json.loads(stub.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}
    data.update({
        "run_id": rid,
        "a": sample["a"],
        "b": sample["b"],
        "E1": sample["E1_MPa"],
        "E2": sample["E2_MPa"],
        "E1_1e11Pa": sample["E1_1e11Pa"],
        "E2_1e11Pa": sample["E2_1e11Pa"],
    })
    if inp_path.exists() and inp_path.stat().st_size > 0:
        data["status"] = "inp_ready"
        data["inp_file"] = inp_path.name
        stub.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return {"run_id": rid, "status": "inp_ready", "rc": rc, "skipped": False}

    data["status"] = "failed"
    data["message"] = data.get("message") or f"generate-inp failed rc={rc}"
    stub.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {"run_id": rid, "status": "failed", "rc": rc, "skipped": False}


def cmd_generate_inp(cfg: dict) -> None:
    try:
        csv_path = resolve_samples_csv(cfg)
    except FileNotFoundError:
        print("Samples CSV missing; running sample first...")
        csv_path = cmd_sample(cfg)
    samples = read_samples(csv_path)
    Path(cfg["runs_dir"]).mkdir(parents=True, exist_ok=True)

    workers = int(cfg.get("gen_inp_workers", 1))
    results = []
    t0 = time.time()
    if workers <= 1:
        for i, s in enumerate(samples):
            r = generate_one(cfg, s)
            results.append(r)
            print(f"[gen {i + 1}/{len(samples)}] run_{r['run_id']:04d} "
                  f"status={r['status']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(generate_one, cfg, s): s for s in samples}
            done = 0
            for fut in as_completed(futs):
                r = fut.result()
                results.append(r)
                done += 1
                print(f"[gen {done}/{len(samples)}] run_{r['run_id']:04d} "
                      f"status={r['status']}", flush=True)

    ok = sum(1 for r in results if r["status"] == "inp_ready")
    write_manifest(cfg, "generate_inp", {
        "ok": ok, "total": len(results), "elapsed_s": time.time() - t0,
        "results": results,
    })
    print(f"generate-inp done: {ok}/{len(results)} ready")


# ----------------------------------------------------------------------
# Stage: solve-parallel
# ----------------------------------------------------------------------
def solve_one(cfg: dict, sample: dict) -> dict:
    rid = int(sample["run_id"])
    rd = run_dir(cfg, rid)
    job = f"run_{rid:04d}"
    inp = rd / f"{job}.inp"
    odb = rd / f"{job}.odb"

    if not inp.exists():
        return {"run_id": rid, "status": "no_inp"}

    # Resume if ODB exists and looks non-empty
    if odb.exists() and odb.stat().st_size > 0:
        sta = rd / f"{job}.sta"
        # Prefer checking .sta for COMPLETED if present
        if sta.exists():
            txt = sta.read_text(encoding="utf-8", errors="replace").upper()
            if "COMPLETED" in txt and "ABORTED" not in txt.split("COMPLETED")[-1][:200]:
                return {"run_id": rid, "status": "solved", "skipped": True}
        else:
            return {"run_id": rid, "status": "solved", "skipped": True}

    cpus = int(cfg.get("cpus_per_job", 4))
    cmd = [
        str(cfg.get("abaqus_cmd", "abaqus")),
        f"job={job}",
        f"input={job}.inp",
        f"cpus={cpus}",
        "interactive",
    ]
    rc = run_cmd(cmd, rd, int(cfg.get("timeout_solve_s", 1800)),
                 rd / "solve.log")

    if odb.exists() and odb.stat().st_size > 0:
        return {"run_id": rid, "status": "solved", "rc": rc, "skipped": False}
    return {"run_id": rid, "status": "failed", "rc": rc, "skipped": False}


def cmd_solve_parallel(cfg: dict, max_workers: int | None = None) -> None:
    samples = read_samples(resolve_samples_csv(cfg))
    workers = max_workers or int(cfg.get("max_workers", 4))
    print(f"solve-parallel: {len(samples)} jobs, max_workers={workers}")

    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(solve_one, cfg, s): s for s in samples}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            done += 1
            print(f"[solve {done}/{len(samples)}] run_{r['run_id']:04d} "
                  f"status={r['status']}", flush=True)

    ok = sum(1 for r in results if r["status"] == "solved")
    write_manifest(cfg, "solve", {
        "ok": ok, "total": len(results), "elapsed_s": time.time() - t0,
        "max_workers": workers, "results": results,
    })
    print(f"solve-parallel done: {ok}/{len(results)} with ODB")


# ----------------------------------------------------------------------
# Stage: extract
# ----------------------------------------------------------------------
def extract_one(cfg: dict, sample: dict) -> dict:
    rid = int(sample["run_id"])
    rd = run_dir(cfg, rid)
    result_file = rd / f"result_{rid:04d}.json"
    odb = rd / f"run_{rid:04d}.odb"

    if result_file.exists():
        try:
            prev = json.loads(result_file.read_text(encoding="utf-8"))
            if prev.get("status") == "ok" and prev.get("frequencies_Hz"):
                return {"run_id": rid, "status": "ok", "skipped": True}
        except json.JSONDecodeError:
            pass

    if not odb.exists():
        return {"run_id": rid, "status": "no_odb"}

    # Ensure extract script is present
    extract_src = Path(cfg.get("script_extract", ROOT / "extract_odb.py"))
    if not extract_src.exists():
        extract_src = ROOT / "extract_odb.py"
    if extract_src.exists():
        shutil.copy(extract_src, rd / "extract_odb.py")

    # Seed JSON with params before extract overwrites status
    seed = {
        "run_id": rid,
        "a": sample["a"],
        "b": sample["b"],
        "E1": sample["E1_MPa"],
        "E2": sample["E2_MPa"],
        "E1_1e11Pa": sample["E1_1e11Pa"],
        "E2_1e11Pa": sample["E2_1e11Pa"],
        "status": "started",
        "frequencies_Hz": [],
        "mode_shapes": {},
    }
    if result_file.exists():
        try:
            seed.update(json.loads(result_file.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            pass
    result_file.write_text(json.dumps(seed, indent=2), encoding="utf-8")

    cmd = [
        str(cfg.get("abaqus_cmd", "abaqus")),
        "python",
        "extract_odb.py",
        "--",
        str(rid),
    ]
    rc = run_cmd(cmd, rd, int(cfg.get("timeout_extract_s", 600)),
                 rd / "extract.log")

    if result_file.exists():
        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
            # keep Table-6 unit fields
            data["E1_1e11Pa"] = sample["E1_1e11Pa"]
            data["E2_1e11Pa"] = sample["E2_1e11Pa"]
            data["E1"] = sample["E1_MPa"]
            data["E2"] = sample["E2_MPa"]
            data["a"] = sample["a"]
            data["b"] = sample["b"]
            result_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return {
                "run_id": rid,
                "status": data.get("status", "unknown"),
                "rc": rc,
                "skipped": False,
            }
        except json.JSONDecodeError:
            pass
    return {"run_id": rid, "status": "failed", "rc": rc, "skipped": False}


def cmd_extract(cfg: dict, max_workers: int | None = None) -> None:
    samples = read_samples(resolve_samples_csv(cfg))
    workers = max_workers or int(cfg.get("max_workers", 4))
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(extract_one, cfg, s): s for s in samples}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            done += 1
            print(f"[extract {done}/{len(samples)}] run_{r['run_id']:04d} "
                  f"status={r['status']}", flush=True)

    ok = sum(1 for r in results if r["status"] == "ok")
    write_manifest(cfg, "extract", {
        "ok": ok, "total": len(results), "elapsed_s": time.time() - t0,
        "results": results,
    })
    print(f"extract done: {ok}/{len(results)} ok")


def cmd_build(cfg: dict) -> None:
    build_script = ROOT / "build_dataset.py"
    cmd = [sys.executable, str(build_script), "--config",
           str(ROOT / "config_dataset.yaml")]
    subprocess.run(cmd, cwd=str(ROOT), check=False)


# ----------------------------------------------------------------------
# Legacy one-shot serial (optional)
# ----------------------------------------------------------------------
def cmd_legacy_full(cfg: dict) -> None:
    """Serial CAE full solve (old behaviour); kept for debugging."""
    try:
        csv_path = resolve_samples_csv(cfg)
    except FileNotFoundError:
        csv_path = cmd_sample(cfg)
    samples = read_samples(csv_path)
    for i, s in enumerate(samples):
        rd = _prepare_run_dir(cfg, s)
        result_file = rd / f"result_{s['run_id']:04d}.json"
        if result_file.exists():
            try:
                prev = json.loads(result_file.read_text(encoding="utf-8"))
                if prev.get("status") == "ok":
                    print(f"[legacy {i + 1}] skip run_{s['run_id']:04d}")
                    continue
            except json.JSONDecodeError:
                pass
        cmd = [
            str(cfg.get("abaqus_cmd", "abaqus")),
            "cae",
            "noGUI=abaqus_run_single.py",
            "--",
            str(s["run_id"]),
            f"{s['a']:.6f}", f"{s['b']:.6f}",
            f"{s['E1_MPa']:.3f}", f"{s['E2_MPa']:.3f}",
            "full",
        ]
        run_cmd(cmd, rd, int(cfg.get("timeout_gen_s", 1800)),
                rd / "legacy_full.log")
        if result_file.exists():
            data = json.loads(result_file.read_text(encoding="utf-8"))
            data["E1_1e11Pa"] = s["E1_1e11Pa"]
            data["E2_1e11Pa"] = s["E2_1e11Pa"]
            result_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"[legacy {i + 1}/{len(samples)}] "
                  f"run_{s['run_id']:04d} status={data.get('status')}")
        else:
            print(f"[legacy {i + 1}/{len(samples)}] "
                  f"run_{s['run_id']:04d} no_result")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="LHS + INP write / parallel solve / extract pipeline")
    ap.add_argument("--config", type=Path,
                    default=ROOT / "config_dataset.yaml")
    ap.add_argument("--n", type=int, default=None,
                    help="override n_samples")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-workers", type=int, default=None)
    ap.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=[
            "sample", "generate-inp", "solve-parallel", "extract", "build",
            "all", "legacy-full",
        ],
        help="pipeline stage (default: all)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.n is not None:
        cfg["n_samples"] = args.n
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.max_workers is not None:
        cfg["max_workers"] = args.max_workers

    cmd = args.command
    if cmd == "sample":
        cmd_sample(cfg)
    elif cmd == "generate-inp":
        cmd_generate_inp(cfg)
    elif cmd == "solve-parallel":
        cmd_solve_parallel(cfg, args.max_workers)
    elif cmd == "extract":
        cmd_extract(cfg, args.max_workers)
    elif cmd == "build":
        cmd_build(cfg)
    elif cmd == "legacy-full":
        cmd_legacy_full(cfg)
    elif cmd == "all":
        cmd_sample(cfg)
        cmd_generate_inp(cfg)
        cmd_solve_parallel(cfg, args.max_workers)
        cmd_extract(cfg, args.max_workers)
        cmd_build(cfg)
    else:
        ap.error(f"unknown command {cmd}")


if __name__ == "__main__":
    # Windows ProcessPoolExecutor needs freeze_support on some setups
    from multiprocessing import freeze_support
    freeze_support()
    main()
