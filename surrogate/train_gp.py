"""
train_gp.py
===========
Train a multi-output Gaussian Process on 7 natural frequencies.

Uses sklearn MultiOutputRegressor(GaussianProcessRegressor) — one independent
GP per mode (standard, robust for N~200–1000 and 4D inputs).

Usage (from repo root or this folder):
    python surrogate/train_gp.py
    python surrogate/train_gp.py --config surrogate/config.yaml
    python surrogate/train_gp.py --npz dataset/train.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_data_path(cfg: dict, override: Path | None) -> Path:
    if override is not None:
        p = override if override.is_absolute() else (Path.cwd() / override)
        if not p.exists():
            p = ROOT / override
        return p.resolve()
    for key in ("npz", "csv"):
        raw = cfg.get("data", {}).get(key)
        if not raw:
            continue
        p = Path(raw)
        if not p.is_absolute():
            p = (HERE / p).resolve()
        if p.exists():
            return p
    raise FileNotFoundError(
        "No training data found. Run the FE pipeline first, e.g.:\n"
        "  python driver.py all --n 200 --seed 42 --max-workers 4\n"
        "then:\n"
        "  python surrogate/train_gp.py"
    )


def load_xy(path: Path, n_modes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        theta = np.asarray(data["theta"], dtype=np.float64)
        freqs = np.asarray(data["freqs_hz"], dtype=np.float64)
        run_id = np.asarray(data["run_id"])
    elif path.suffix.lower() == ".csv":
        import csv
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        theta = np.array(
            [[float(r["a_mm"]), float(r["b_mm"]),
              float(r["E1_1e11Pa"]), float(r["E2_1e11Pa"])] for r in rows],
            dtype=np.float64,
        )
        freqs = np.array(
            [[float(r[f"f{i}_Hz"]) for i in range(1, n_modes + 1)] for r in rows],
            dtype=np.float64,
        )
        run_id = np.array([int(r["run_id"]) for r in rows], dtype=np.int32)
    else:
        raise ValueError(f"Unsupported data file: {path}")

    if freqs.ndim != 2 or freqs.shape[1] < n_modes:
        raise ValueError(
            f"Expected freqs shape (N,>={n_modes}), got {freqs.shape}")
    freqs = freqs[:, :n_modes]

    # Drop rows with NaN
    mask = np.isfinite(theta).all(axis=1) & np.isfinite(freqs).all(axis=1)
    if not mask.any():
        raise ValueError("All rows have NaN; check dataset")
    return theta[mask], freqs[mask], run_id[mask]


def make_gp(cfg: dict) -> MultiOutputRegressor:
    gp_cfg = cfg.get("gp", {})
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=np.ones(4), length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-1))
    )
    base = GaussianProcessRegressor(
        kernel=kernel,
        alpha=float(gp_cfg.get("alpha", 1e-8)),
        normalize_y=bool(gp_cfg.get("normalize_y", True)),
        n_restarts_optimizer=int(gp_cfg.get("n_restarts_optimizer", 5)),
        random_state=int(cfg.get("seed", 42)),
    )
    return MultiOutputRegressor(base, n_jobs=-1)


def metrics_report(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, Any]:
    err = y_pred - y_true
    abs_err = np.abs(err)
    rel = abs_err / np.maximum(np.abs(y_true), 1e-12)
    per_mode = []
    for i in range(y_true.shape[1]):
        per_mode.append({
            "mode": i + 1,
            "mae_Hz": float(np.mean(abs_err[:, i])),
            "rmse_Hz": float(np.sqrt(np.mean(err[:, i] ** 2))),
            "mape": float(np.mean(rel[:, i])),
            "max_abs_Hz": float(np.max(abs_err[:, i])),
        })
    return {
        "n_test": int(y_true.shape[0]),
        "mae_Hz_mean": float(np.mean(abs_err)),
        "rmse_Hz_mean": float(np.sqrt(np.mean(err ** 2))),
        "mape_mean": float(np.mean(rel)),
        "per_mode": per_mode,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train multi-output GP surrogate")
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--npz", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    n_modes = int(cfg.get("n_modes", 7))
    out_dir = args.out_dir or HERE / cfg.get("out_dir", ".")
    out_dir = out_dir if out_dir.is_absolute() else (HERE / out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = resolve_data_path(cfg, args.npz)
    print(f"Loading data: {data_path}")
    X, Y, run_id = load_xy(data_path, n_modes)
    print(f"Loaded N={len(X)} samples, n_modes={n_modes}")

    if len(X) < 10:
        raise SystemExit(
            f"Need more successful FE samples (got {len(X)}). "
            "Finish driver generate-inp / solve / extract / build first."
        )

    test_frac = float(cfg.get("test_fraction", 0.2))
    seed = int(cfg.get("seed", 42))
    X_tr, X_te, Y_tr, Y_te, id_tr, id_te = train_test_split(
        X, Y, run_id, test_size=test_frac, random_state=seed
    )

    x_scaler = StandardScaler()
    X_tr_s = x_scaler.fit_transform(X_tr)
    X_te_s = x_scaler.transform(X_te)

    model = make_gp(cfg)
    print(f"Fitting multi-output GP on {len(X_tr)} train / {len(X_te)} test ...")
    model.fit(X_tr_s, Y_tr)

    Y_hat = model.predict(X_te_s)
    metrics = metrics_report(Y_te, Y_hat)
    metrics.update({
        "n_train": int(len(X_tr)),
        "n_total": int(len(X)),
        "data_path": str(data_path),
        "test_run_ids": id_te.tolist(),
        "theta_columns": cfg.get(
            "theta_columns",
            ["a_mm", "b_mm", "E1_1e11Pa", "E2_1e11Pa"],
        ),
    })

    model_path = out_dir / cfg.get("model_file", "multioutput_gp.joblib")
    scaler_path = out_dir / cfg.get("scaler_file", "input_scaler.joblib")
    metrics_path = out_dir / cfg.get("metrics_file", "metrics.json")

    joblib.dump(
        {
            "model": model,
            "n_modes": n_modes,
            "theta_columns": metrics["theta_columns"],
        },
        model_path,
    )
    joblib.dump(x_scaler, scaler_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Also save a small prediction table for the test set
    pred_path = out_dir / "test_predictions.csv"
    header = ["run_id"] + [f"f{i}_true" for i in range(1, n_modes + 1)] + [
        f"f{i}_pred" for i in range(1, n_modes + 1)
    ]
    lines = [",".join(header)]
    for i in range(len(id_te)):
        row = [str(int(id_te[i]))]
        row += [f"{v:.8g}" for v in Y_te[i]]
        row += [f"{v:.8g}" for v in Y_hat[i]]
        lines.append(",".join(row))
    pred_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved model   -> {model_path}")
    print(f"Saved scaler  -> {scaler_path}")
    print(f"Saved metrics -> {metrics_path}")
    print(f"Test MAPE (mean over modes): {metrics['mape_mean'] * 100:.3f}%")
    print(f"Test RMSE (mean over modes): {metrics['rmse_Hz_mean']:.4f} Hz")
    for m in metrics["per_mode"]:
        print(
            f"  mode {m['mode']}: MAE={m['mae_Hz']:.4f} Hz  "
            f"MAPE={m['mape'] * 100:.3f}%"
        )

    # Parity / R^2 figures
    try:
        import sys
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        from plot_r2 import compute_r2, plot_parity, plot_r2_bar
        import json as _json
        r2_info = compute_r2(Y_te, Y_hat)
        plot_parity(Y_te, Y_hat, r2_info, out_dir / "parity_r2.png")
        plot_r2_bar(r2_info, out_dir / "r2_by_mode.png")
        (out_dir / "r2_scores.json").write_text(
            _json.dumps(r2_info, indent=2), encoding="utf-8")
        metrics["r2_mean"] = r2_info["r2_mean"]
        metrics["r2_min"] = r2_info["r2_min"]
        r2_map = {d["mode"]: d["r2"] for d in r2_info["per_mode_r2"]}
        for m in metrics["per_mode"]:
            m["r2"] = r2_map.get(m["mode"])
        metrics_path.write_text(_json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Test R2 (mean over modes): {r2_info['r2_mean']:.6f}")
        print(f"Saved parity plot -> {out_dir / 'parity_r2.png'}")
    except Exception as exc:
        print(f"Warning: R2 plots not generated ({exc})")


if __name__ == "__main__":
    main()
