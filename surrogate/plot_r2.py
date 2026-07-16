"""
plot_r2.py
==========
Compute per-mode R^2 and save parity (true vs predicted) plots.

Reads surrogate/test_predictions.csv by default (written by train_gp.py).

Usage:
    python surrogate/plot_r2.py
    python surrogate/plot_r2.py --csv surrogate/test_predictions.csv --out-dir surrogate
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score


HERE = Path(__file__).resolve().parent


def load_predictions(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"Empty predictions CSV: {csv_path}")

    n_modes = 0
    while f"f{n_modes + 1}_true" in rows[0]:
        n_modes += 1
    if n_modes == 0:
        raise ValueError("No f*_true columns found")

    run_id = np.array([int(r["run_id"]) for r in rows], dtype=np.int32)
    y_true = np.array(
        [[float(r[f"f{i}_true"]) for i in range(1, n_modes + 1)] for r in rows],
        dtype=np.float64,
    )
    y_pred = np.array(
        [[float(r[f"f{i}_pred"]) for i in range(1, n_modes + 1)] for r in rows],
        dtype=np.float64,
    )
    return run_id, y_true, y_pred


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    n_modes = y_true.shape[1]
    per_mode = []
    r2_list = []
    for i in range(n_modes):
        r2 = float(r2_score(y_true[:, i], y_pred[:, i]))
        r2_list.append(r2)
        per_mode.append({"mode": i + 1, "r2": r2})
    return {
        "r2_mean": float(np.mean(r2_list)),
        "r2_min": float(np.min(r2_list)),
        "per_mode_r2": per_mode,
    }


def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    r2_info: dict,
    out_path: Path,
) -> None:
    n_modes = y_true.shape[1]
    ncols = 4
    nrows = int(np.ceil(n_modes / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for i in range(n_modes):
        ax = axes[i]
        yt, yp = y_true[:, i], y_pred[:, i]
        r2 = r2_info["per_mode_r2"][i]["r2"]
        ax.scatter(yt, yp, s=22, alpha=0.75, edgecolors="none", c="#1f4e79")
        lo = float(min(yt.min(), yp.min()))
        hi = float(max(yt.max(), yp.max()))
        pad = 0.03 * (hi - lo + 1e-12)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.0)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"True $f_{i + 1}$ [Hz]")
        ax.set_ylabel(f"Pred $f_{i + 1}$ [Hz]")
        ax.set_title(f"Mode {i + 1}: $R^2$ = {r2:.4f}")
        ax.grid(True, alpha=0.25)

    for j in range(n_modes, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Multi-output GP parity (test set)  |  mean $R^2$ = {r2_info['r2_mean']:.4f}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_r2_bar(r2_info: dict, out_path: Path) -> None:
    modes = [d["mode"] for d in r2_info["per_mode_r2"]]
    vals = [d["r2"] for d in r2_info["per_mode_r2"]]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(modes, vals, color="#2a6f97", edgecolor="none")
    ax.axhline(r2_info["r2_mean"], color="#c1121f", ls="--", lw=1.2,
               label=f"mean = {r2_info['r2_mean']:.4f}")
    ax.set_xlabel("Mode")
    ax.set_ylabel(r"$R^2$")
    ax.set_ylim(min(0.95, min(vals) - 0.01), 1.001)
    ax.set_xticks(modes)
    ax.set_title("Per-mode $R^2$ on hold-out test set")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.0005, f"{v:.4f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot R^2 / parity for GP surrogate")
    ap.add_argument("--csv", type=Path, default=HERE / "test_predictions.csv")
    ap.add_argument("--out-dir", type=Path, default=HERE)
    ap.add_argument("--metrics", type=Path, default=HERE / "metrics.json",
                    help="optional metrics.json to update with R2 fields")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    _, y_true, y_pred = load_predictions(args.csv)
    r2_info = compute_r2(y_true, y_pred)

    parity_path = out_dir / "parity_r2.png"
    bar_path = out_dir / "r2_by_mode.png"
    r2_json_path = out_dir / "r2_scores.json"

    plot_parity(y_true, y_pred, r2_info, parity_path)
    plot_r2_bar(r2_info, bar_path)
    r2_json_path.write_text(json.dumps(r2_info, indent=2), encoding="utf-8")

    if args.metrics.exists():
        metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
        metrics["r2_mean"] = r2_info["r2_mean"]
        metrics["r2_min"] = r2_info["r2_min"]
        # merge into per_mode if present
        if "per_mode" in metrics:
            r2_map = {d["mode"]: d["r2"] for d in r2_info["per_mode_r2"]}
            for m in metrics["per_mode"]:
                m["r2"] = r2_map.get(m["mode"])
        else:
            metrics["per_mode_r2"] = r2_info["per_mode_r2"]
        args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"R2 mean = {r2_info['r2_mean']:.6f}")
    for d in r2_info["per_mode_r2"]:
        print(f"  mode {d['mode']}: R2 = {d['r2']:.6f}")
    print(f"Saved {parity_path}")
    print(f"Saved {bar_path}")
    print(f"Saved {r2_json_path}")


if __name__ == "__main__":
    main()
