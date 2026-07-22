"""
infer_obs.py -- pooled posterior inference over a batch of observed
frequency vectors.

For each method, ALL observations' posterior samples are pooled into one
combined set, saved as a single CSV, and shown in a single pair plot.
(The pooled distribution approximates the population posterior over the
30 test structures, i.e. the parameter-variability recovery target.)

Expected observation file (default): data/obs_natural_frequencies_10.csv
    header row, then rows: index, f1..f10 [Hz]  -- only f1..f5 are used.
Optional truth file: data/obs_input.csv (no header, logged parameters per
row, e.g. a, b) -- overlaid on the pair plots and used for the recovery
plot when present.

Usage:
    python infer_obs.py --method all --n 5000
    python infer_obs.py --method cfm --obs-csv data\\obs_natural_frequencies_10.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import config as C
from data import Standardiser
from models import REGISTRY


# ----------------------------------------------------------------------
def _numeric_rows(path: Path) -> list[list[float]]:
    rows = []
    with path.open() as f:
        for line in f:
            vals = []
            for p in line.replace(";", ",").split(","):
                try:
                    vals.append(float(p.strip()))
                except ValueError:
                    pass
            if vals:
                rows.append(vals)
    return rows


def read_obs(path: Path) -> np.ndarray:
    """Rows may be [index, f1..fK] or [f1..fK]; keep the first Y_DIM
    frequencies."""
    out = []
    for vals in _numeric_rows(path):
        if len(vals) > C.Y_DIM:          # leading order/index column
            out.append(vals[1:C.Y_DIM + 1])
        elif len(vals) == C.Y_DIM:
            out.append(vals)
    return np.asarray(out, dtype=np.float64)


def read_truth(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    rows = _numeric_rows(path)
    return np.asarray(rows, dtype=np.float64) if rows else None


def load_model(method: str, ckpt_dir: Path, device):
    ck = torch.load(ckpt_dir / f"{method}.pt", map_location=device,
                    weights_only=False)
    model = REGISTRY[method](C.THETA_DIM, C.Y_DIM,
                             ck["common_cfg"], ck["method_cfg"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, Standardiser.from_state(ck["s_theta"]), \
        Standardiser.from_state(ck["s_y"])


# ----------------------------------------------------------------------
def pooled_pair_plot(samples: np.ndarray, truth: np.ndarray | None,
                     title: str, out: Path):
    """samples: [N, 4] pooled over all observations; truth: [N_obs, n_t]
    scattered on the panels that have logged values."""
    D = samples.shape[1]
    names = C.PARAM_NAMES
    n_t = truth.shape[1] if truth is not None else 0
    fig, axes = plt.subplots(D, D, figsize=(2.4 * D, 2.4 * D))
    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                ax.hist(samples[:, i], bins=70, color="#4878a8",
                        density=True)
                if i < n_t:
                    for v in truth[:, i]:
                        ax.axvline(v, color="crimson", lw=0.5, alpha=0.35)
            else:
                ax.hist2d(samples[:, j], samples[:, i], bins=80,
                          cmap="Blues")
                if j < n_t and i < n_t:
                    ax.scatter(truth[:, j], truth[:, i], s=14,
                               color="crimson", marker="x", lw=1.0,
                               label="logged truth")
            if i == D - 1:
                ax.set_xlabel(names[j])
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(names[i])
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def recovery_plot(means, lo, hi, truth, method: str, out: Path):
    n_true = truth.shape[1]
    fig, axes = plt.subplots(1, n_true, figsize=(4.5 * n_true, 4))
    if n_true == 1:
        axes = [axes]
    for j in range(n_true):
        ax = axes[j]
        t, m = truth[:, j], means[:, j]
        ax.errorbar(t, m, yerr=[m - lo[:, j], hi[:, j] - m], fmt="o",
                    ms=4, lw=0.8, capsize=2, color="#4878a8",
                    ecolor="#9db8d2")
        lim = [min(t.min(), m.min()), max(t.max(), m.max())]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel(f"true {C.PARAM_NAMES[j]}")
        ax.set_ylabel(f"posterior mean {C.PARAM_NAMES[j]}")
        ax.set_title(f"{method}: {C.PARAM_NAMES[j]} recovery")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=[*REGISTRY.keys(), "all"])
    ap.add_argument("--obs-csv", type=Path,
                    default=Path("data/obs_natural_frequencies_10.csv"))
    ap.add_argument("--input-csv", type=Path,
                    default=Path("data/obs_input.csv"))
    ap.add_argument("--n", type=int, default=5000,
                    help="posterior samples PER observation")
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_all = read_obs(args.obs_csv)
    truth = read_truth(args.input_csv)
    print(f"observations: {y_all.shape} (first {C.Y_DIM} modes used)")
    if truth is not None:
        if len(truth) != len(y_all):
            print(f"WARNING: truth rows ({len(truth)}) != obs rows "
                  f"({len(y_all)}); truth overlay disabled")
            truth = None
        else:
            print(f"logged inputs: {truth.shape}")

    methods = list(REGISTRY) if args.method == "all" else [args.method]
    args.out.mkdir(parents=True, exist_ok=True)

    for m in methods:
        model, s_theta, s_y = load_model(m, args.ckpt, device)
        y_std = torch.from_numpy(
            s_y.transform(y_all).astype(np.float32)).to(device)
        with torch.no_grad():
            th = model.sample(y_std, args.n)          # [n, N_obs, 4]
        th = th.cpu().numpy()
        N_obs = th.shape[1]
        th_phys = s_theta.inverse(
            th.reshape(-1, C.THETA_DIM)).reshape(th.shape)
        # shift 'a' into the physical/logged-truth frame (wing-root datum)
        th_phys = C.to_physical_frame(th_phys)

        # ---- combined CSV: obs_id + parameter columns -----------------
        pooled = th_phys.transpose(1, 0, 2).reshape(-1, C.THETA_DIM)
        obs_id = np.repeat(np.arange(N_obs), args.n)
        csv_path = args.out / f"samples_{m}.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["obs_id", *C.PARAM_NAMES])
            for k in range(len(pooled)):
                w.writerow([int(obs_id[k]),
                            *[f"{v:.6f}" for v in pooled[k]]])

        # ---- pooled pair plot -----------------------------------------
        pooled_pair_plot(
            pooled, truth,
            f"{m} | pooled posterior over {N_obs} observations "
            f"({args.n} samples each)",
            args.out / f"pairplot_{m}.png")

        # ---- recovery + coverage (needs truth) ------------------------
        if truth is not None:
            means = th_phys.mean(axis=0)
            lo = np.percentile(th_phys, 5, axis=0)
            hi = np.percentile(th_phys, 95, axis=0)
            n_t = truth.shape[1]
            recovery_plot(means[:, :n_t], lo[:, :n_t], hi[:, :n_t],
                          truth, m, args.out / f"recovery_{m}.png")
            cov = [np.mean([(lo[k, j] <= truth[k, j] <= hi[k, j])
                            for k in range(N_obs)]) for j in range(n_t)]
            print(f"[{m}] 90% CI empirical coverage: "
                  + ", ".join(f"{C.PARAM_NAMES[j]}={cov[j]:.2f}"
                              for j in range(n_t)))
        print(f"[{m}] -> {csv_path}, pairplot_{m}.png"
              + ("" if truth is None else f", recovery_{m}.png"))


if __name__ == "__main__":
    main()
