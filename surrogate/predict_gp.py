"""
predict_gp.py
=============
Predict 7 natural frequencies with the trained multi-output GP.

Examples:
    python surrogate/predict_gp.py --a 300 --b 25 --E1 0.6 --E2 0.7
    python surrogate/predict_gp.py --csv samples/lhs_5_seed42.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np


HERE = Path(__file__).resolve().parent


def load_bundle(model_dir: Path):
    bundle = joblib.load(model_dir / "multioutput_gp.joblib")
    scaler = joblib.load(model_dir / "input_scaler.joblib")
    return bundle["model"], scaler, int(bundle.get("n_modes", 7))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=HERE)
    ap.add_argument("--a", type=float, default=None, help="mm")
    ap.add_argument("--b", type=float, default=None, help="mm")
    ap.add_argument("--E1", type=float, default=None, help="x 1e11 Pa")
    ap.add_argument("--E2", type=float, default=None, help="x 1e11 Pa")
    ap.add_argument("--csv", type=Path, default=None,
                    help="CSV with a,b,E1_1e11Pa,E2_1e11Pa columns")
    ap.add_argument("--return-std", action="store_true",
                    help="also print predictive std (per estimator)")
    args = ap.parse_args()

    model, scaler, n_modes = load_bundle(args.model_dir)

    if args.csv is not None:
        rows = list(csv.DictReader(args.csv.open(encoding="utf-8")))
        X = []
        for r in rows:
            if "E1_1e11Pa" in r:
                e1, e2 = float(r["E1_1e11Pa"]), float(r["E2_1e11Pa"])
            else:
                e1, e2 = float(r["E1"]), float(r["E2"])
            X.append([float(r["a"]), float(r["b"]), e1, e2])
        X = np.asarray(X, dtype=np.float64)
    else:
        if None in (args.a, args.b, args.E1, args.E2):
            raise SystemExit("Provide --a --b --E1 --E2 or --csv")
        X = np.array([[args.a, args.b, args.E1, args.E2]], dtype=np.float64)

    Xs = scaler.transform(X)
    mean = model.predict(Xs)

    if args.return_std:
        # Independent GPs expose predict(return_std) on each estimator_
        stds = []
        for est in model.estimators_:
            _, s = est.predict(Xs, return_std=True)
            stds.append(s)
        std = np.column_stack(stds)
        for i in range(len(X)):
            print(f"sample[{i}] theta={X[i].tolist()}")
            for m in range(n_modes):
                print(f"  f{m + 1} = {mean[i, m]:.6f} ± {std[i, m]:.6f} Hz")
    else:
        for i in range(len(X)):
            print(f"sample[{i}] theta={X[i].tolist()}")
            for m in range(n_modes):
                print(f"  f{m + 1} = {mean[i, m]:.6f} Hz")


if __name__ == "__main__":
    main()
