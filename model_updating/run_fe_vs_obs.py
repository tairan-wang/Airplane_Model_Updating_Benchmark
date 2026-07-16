"""
run_fe_vs_obs.py
================
For each (a, b) in obs_input.csv, draw one LHS (E1, E2) in Table-6 bounds,
run the FE pipeline, and scatter-plot FE vs experimental natural frequencies
(first 7 modes).

Usage:
    python model_updating/run_fe_vs_obs.py              # sample + FE + plot
    python model_updating/run_fe_vs_obs.py --sample-only
    python model_updating/run_fe_vs_obs.py --plot-only   # after FE finished
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
BIAS = HERE / "fe_bias"
DATA = HERE / "data"
CFG = BIAS / "config.yaml"
DRIVER = ROOT / "driver.py"

OBS_INPUT = DATA / "obs_input.csv"
OBS_FREQ = DATA / "obs_natural_frequencies_10.csv"
SAMPLES_CSV = BIAS / "obs_lhs_E_seed7.csv"
N_MODES = 7
E_LO, E_HI = 0.5, 0.9  # x 1e11 Pa


def e_to_mpa(e_1e11: float) -> float:
    return float(e_1e11) * 1e5


def load_obs_ab(path: Path) -> np.ndarray:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b = line.split(",")[:2]
            rows.append([float(a), float(b)])
    return np.asarray(rows, dtype=np.float64)


def load_obs_freqs(path: Path, n_modes: int = N_MODES) -> np.ndarray:
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        freqs = []
        for row in reader:
            freqs.append([float(row[f"f{i}_Hz"]) for i in range(1, n_modes + 1)])
    return np.asarray(freqs, dtype=np.float64)


def write_samples(ab: np.ndarray, seed: int) -> Path:
    n = len(ab)
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    u = sampler.random(n=n)
    e = qmc.scale(u, [E_LO, E_LO], [E_HI, E_HI])

    BIAS.mkdir(parents=True, exist_ok=True)
    with SAMPLES_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "run_id", "a", "b",
            "E1_1e11Pa", "E2_1e11Pa", "E1_MPa", "E2_MPa",
            "obs_row",
        ])
        for i in range(n):
            e1, e2 = e[i]
            w.writerow([
                i,
                f"{ab[i, 0]:.8g}", f"{ab[i, 1]:.8g}",
                f"{e1:.8g}", f"{e2:.8g}",
                f"{e_to_mpa(e1):.8g}", f"{e_to_mpa(e2):.8g}",
                i + 1,
            ])

    # Point driver at this CSV
    (BIAS / "active_samples.txt").write_text(
        str(SAMPLES_CSV.resolve()), encoding="utf-8")
    print(f"Wrote {SAMPLES_CSV}  (N={n}, seed={seed})")
    return SAMPLES_CSV


def run_fe_stages() -> None:
    # build uses root config; we assemble freqs from result JSON in plot step
    stages = ["generate-inp", "solve-parallel", "extract"]
    for stage in stages:
        cmd = [
            sys.executable, str(DRIVER),
            "--config", str(CFG),
            "--max-workers", "4",
            stage,
        ]
        print(">>", " ".join(cmd), flush=True)
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            raise SystemExit(f"Stage {stage} failed with code {rc}")


def load_fe_freqs(n: int, n_modes: int = N_MODES) -> tuple[np.ndarray, list[str]]:
    """Return (N, n_modes) array; NaN if missing/failed. Also status list."""
    runs = BIAS / "runs"
    freqs = np.full((n, n_modes), np.nan, dtype=np.float64)
    statuses = []
    for i in range(n):
        p = runs / f"run_{i:04d}" / f"result_{i:04d}.json"
        if not p.exists():
            statuses.append("missing")
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        st = data.get("status", "unknown")
        statuses.append(st)
        f = data.get("frequencies_Hz") or []
        for j in range(min(n_modes, len(f))):
            freqs[i, j] = float(f[j])
    return freqs, statuses


def plot_compare(obs_f: np.ndarray, fe_f: np.ndarray, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_modes = obs_f.shape[1]
    # Flatten for scatter: x = mode index (with jitter), colour = source
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    # --- left: per-mode overlay (obs vs FE), jittered by sample ---
    ax = axes[0]
    rng = np.random.default_rng(0)
    for i in range(n_modes):
        mode_x = i + 1
        n = obs_f.shape[0]
        jit_o = rng.uniform(-0.12, -0.02, size=n)
        jit_f = rng.uniform(0.02, 0.12, size=n)
        ax.scatter(
            np.full(n, mode_x) + jit_o, obs_f[:, i],
            s=28, c="#c1121f", alpha=0.75, label="Obs (exp)" if i == 0 else None,
            edgecolors="none", zorder=3,
        )
        mask = np.isfinite(fe_f[:, i])
        ax.scatter(
            np.full(mask.sum(), mode_x) + jit_f[mask], fe_f[mask, i],
            s=28, c="#1d3557", alpha=0.75, label="FE (Abaqus)" if i == 0 else None,
            edgecolors="none", zorder=3,
        )
    ax.set_xticks(range(1, n_modes + 1))
    ax.set_xlabel("Mode")
    ax.set_ylabel("Natural frequency [Hz]")
    ax.set_title("Obs vs FE frequencies (7 modes)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # --- right: parity obs vs FE (paired by row) ---
    ax = axes[1]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_modes))
    for i in range(n_modes):
        mask = np.isfinite(fe_f[:, i])
        ax.scatter(
            obs_f[mask, i], fe_f[mask, i],
            s=32, c=[colors[i]], alpha=0.8,
            label=f"f{i + 1}", edgecolors="none",
        )
    all_o = obs_f[np.isfinite(fe_f)]
    all_f = fe_f[np.isfinite(fe_f)]
    if all_o.size and all_f.size:
        lo = float(min(all_o.min(), all_f.min()))
        hi = float(max(all_o.max(), all_f.max()))
        pad = 0.04 * (hi - lo + 1e-12)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.0)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Obs frequency [Hz]")
    ax.set_ylabel("FE frequency [Hz]")
    ax.set_title("Parity (same obs row ↔ FE with LHS E)")
    ax.legend(ncol=2, fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    out = out_dir / "fe_vs_obs_freqs.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_bias_bars(obs_f: np.ndarray, fe_f: np.ndarray, out_dir: Path) -> Path:
    """Mean relative bias (FE - Obs) / Obs per mode."""
    rel = (fe_f - obs_f) / np.maximum(np.abs(obs_f), 1e-12)
    means, stds = [], []
    for i in range(obs_f.shape[1]):
        m = np.isfinite(fe_f[:, i])
        means.append(float(np.nanmean(rel[m, i])) if m.any() else float("nan"))
        stds.append(float(np.nanstd(rel[m, i])) if m.any() else float("nan"))
    modes = np.arange(1, len(means) + 1)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(modes, np.array(means) * 100, yerr=np.array(stds) * 100,
           color="#457b9d", ecolor="#1d3557", capsize=3, alpha=0.9)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("Mode")
    ax.set_ylabel("Relative bias (FE−Obs)/Obs [%]")
    ax.set_title("Mean FE–Obs frequency bias (±1 std over 30 samples)")
    ax.set_xticks(modes)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = out_dir / "fe_vs_obs_bias.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def write_summary(obs_f: np.ndarray, fe_f: np.ndarray, statuses: list[str],
                  out_dir: Path) -> Path:
    n_modes = obs_f.shape[1]
    path = out_dir / "fe_vs_obs_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["obs_row", "status"]
        for i in range(1, n_modes + 1):
            header += [f"obs_f{i}", f"fe_f{i}", f"err_f{i}", f"rel_err_f{i}"]
        w.writerow(header)
        for i in range(len(obs_f)):
            row = [i + 1, statuses[i]]
            for j in range(n_modes):
                o = obs_f[i, j]
                fe = fe_f[i, j]
                if np.isfinite(fe):
                    err = fe - o
                    rel = err / max(abs(o), 1e-12)
                else:
                    err = rel = float("nan")
                row += [f"{o:.8g}", f"{fe:.8g}" if np.isfinite(fe) else "",
                        f"{err:.8g}" if np.isfinite(err) else "",
                        f"{rel:.8g}" if np.isfinite(rel) else ""]
            w.writerow(row)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sample-only", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--skip-fe", action="store_true",
                    help="sample + plot using existing FE results only")
    args = ap.parse_args()

    ab = load_obs_ab(OBS_INPUT)
    obs_f = load_obs_freqs(OBS_FREQ, N_MODES)
    if len(ab) != len(obs_f):
        raise SystemExit(
            f"Row count mismatch: obs_input={len(ab)} vs freqs={len(obs_f)}")

    if not args.plot_only:
        write_samples(ab, seed=args.seed)
        if args.sample_only:
            return
        if not args.skip_fe:
            run_fe_stages()

    fe_f, statuses = load_fe_freqs(len(ab), N_MODES)
    n_ok = sum(1 for s in statuses if s == "ok")
    print(f"FE results: {n_ok}/{len(statuses)} status=ok")

    out_dir = BIAS / "plots"
    p1 = plot_compare(obs_f, fe_f, out_dir)
    p2 = plot_bias_bars(obs_f, fe_f, out_dir)
    p3 = write_summary(obs_f, fe_f, statuses, out_dir)
    print(f"Saved {p1}")
    print(f"Saved {p2}")
    print(f"Saved {p3}")

    # Print mean relative bias
    rel = (fe_f - obs_f) / np.maximum(np.abs(obs_f), 1e-12)
    print("Mean relative bias (FE-Obs)/Obs per mode:")
    for i in range(N_MODES):
        m = np.isfinite(fe_f[:, i])
        if m.any():
            print(f"  f{i + 1}: {100 * np.nanmean(rel[m, i]):+.2f}% "
                  f"(std {100 * np.nanstd(rel[m, i]):.2f}%)")


if __name__ == "__main__":
    main()
