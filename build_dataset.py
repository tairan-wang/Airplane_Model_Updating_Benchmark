"""
build_dataset.py
================
Aggregate successful result_XXXX.json files into dataset/train_N.npz
(and optional CSV) for generative-model training.

Usage:
    python build_dataset.py
    python build_dataset.py --config config_dataset.yaml
    python build_dataset.py --runs-dir samples/runs --out dataset/train.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_result(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("status") != "ok":
        return None
    freqs = data.get("frequencies_Hz") or []
    if not freqs:
        return None
    return data


def shapes_to_array(mode_shapes: dict, n_modes: int, n_sensors: int) -> np.ndarray:
    out = np.full((n_modes, n_sensors), np.nan, dtype=np.float64)
    for i in range(n_modes):
        key = f"mode_{i + 1}"
        comps = mode_shapes.get(key)
        if not comps:
            continue
        arr = np.asarray(comps, dtype=np.float64)
        n = min(n_sensors, arr.size)
        out[i, :n] = arr[:n]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build train.npz from result JSON")
    ap.add_argument("--config", type=Path, default=Path("config_dataset.yaml"))
    ap.add_argument("--runs-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-modes", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config.exists() else {}
    runs_dir = args.runs_dir or Path(cfg.get("runs_dir", "samples/runs"))
    dataset_dir = Path(cfg.get("dataset_dir", "dataset"))
    n_modes = args.n_modes or int(cfg.get("n_modes_keep", 7))
    out_path = args.out or (dataset_dir / "train.npz")

    results = []
    failed = []
    for p in sorted(runs_dir.glob("run_*/result_*.json")):
        data = load_result(p)
        if data is None:
            failed.append(p.parent.name)
            continue
        results.append(data)

    if not results:
        raise SystemExit(f"No successful results under {runs_dir}")

    n_ok = len(results)
    # Infer sensor count from first sample
    first_shapes = results[0].get("mode_shapes") or {}
    n_sensors = len(next(iter(first_shapes.values()), [])) or 5

    run_ids = np.zeros(n_ok, dtype=np.int32)
    theta = np.zeros((n_ok, 4), dtype=np.float64)
    freqs = np.full((n_ok, n_modes), np.nan, dtype=np.float64)
    modes = np.full((n_ok, n_modes, n_sensors), np.nan, dtype=np.float64)

    for i, r in enumerate(results):
        run_ids[i] = int(r["run_id"])
        a = float(r["a"])
        b = float(r["b"])
        e1_pa = r.get("E1_1e11Pa")
        e2_pa = r.get("E2_1e11Pa")
        if e1_pa is None:
            e1_pa = float(r["E1"]) / 1e5  # MPa -> x1e11 Pa
        if e2_pa is None:
            e2_pa = float(r["E2"]) / 1e5
        theta[i] = [a, b, float(e1_pa), float(e2_pa)]

        f = list(r.get("frequencies_Hz") or [])
        n = min(n_modes, len(f))
        freqs[i, :n] = f[:n]
        modes[i] = shapes_to_array(r.get("mode_shapes") or {}, n_modes, n_sensors)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "n_ok": n_ok,
        "n_failed_or_incomplete": len(failed),
        "failed_runs": failed,
        "theta_columns": ["a_mm", "b_mm", "E1_1e11Pa", "E2_1e11Pa"],
        "n_modes": n_modes,
        "n_sensors": n_sensors,
        "bounds": cfg.get("bounds"),
        "seed": cfg.get("seed"),
        "n_samples_requested": cfg.get("n_samples"),
    }
    np.savez_compressed(
        out_path,
        theta=theta,
        freqs_hz=freqs,
        mode_shapes=modes,
        run_id=run_ids,
        meta=np.array([json.dumps(meta)], dtype=object),
    )

    # Also write a flat CSV for quick inspection
    csv_path = out_path.with_suffix(".csv")
    header = ["run_id", "a_mm", "b_mm", "E1_1e11Pa", "E2_1e11Pa"]
    header += [f"f{i}_Hz" for i in range(1, n_modes + 1)]
    lines = [",".join(header)]
    for i in range(n_ok):
        row = [str(run_ids[i])] + [f"{x:.8g}" for x in theta[i]]
        row += [f"{x:.8g}" if np.isfinite(x) else "" for x in freqs[i]]
        lines.append(",".join(row))
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "train_npz": str(out_path),
        "train_csv": str(csv_path),
        "n_ok": n_ok,
        "failed_runs": failed,
        "meta": meta,
    }
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {out_path}  ({n_ok} ok, {len(failed)} skipped)")
    print(f"Wrote {csv_path}")
    print(f"Wrote {dataset_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
