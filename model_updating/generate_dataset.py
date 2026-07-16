"""
generate_dataset.py -- draw theta from the prior, push through the trained
multi-output GP surrogate, keep the first 5 modes, optionally add
measurement-like noise, and save (theta, y) pairs for generative training.

Usage:
    python generate_dataset.py --n 20000 --gp-dir ..\\surrogate
    python generate_dataset.py --n 20000 --gp-dir ..\\surrogate --seed 1 --sobol
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

import config as C


def sample_prior(n: int, seed: int, sobol: bool) -> np.ndarray:
    lo = np.array([C.PRIOR_BOUNDS[k][0] for k in C.PARAM_NAMES])
    hi = np.array([C.PRIOR_BOUNDS[k][1] for k in C.PARAM_NAMES])
    if sobol:
        from scipy.stats import qmc
        eng = qmc.Sobol(d=len(lo), scramble=True, seed=seed)
        m = int(np.ceil(np.log2(max(n, 2))))
        u = eng.random_base2(m=m)[:n]
    else:
        rng = np.random.default_rng(seed)
        u = rng.uniform(size=(n, len(lo)))
    return lo + u * (hi - lo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--gp-dir", type=Path, required=True,
                    help="folder containing multioutput_gp.joblib and "
                         "input_scaler.joblib")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sobol", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("data/dataset.npz"))
    args = ap.parse_args()

    bundle = joblib.load(args.gp_dir / "multioutput_gp.joblib")
    scaler = joblib.load(args.gp_dir / "input_scaler.joblib")
    gp = bundle["model"]

    theta = sample_prior(args.n, args.seed, args.sobol)
    y_full = gp.predict(scaler.transform(theta))
    y = np.asarray(y_full)[:, : C.Y_DIM]              # first 5 modes only

    if C.Y_NOISE_FRAC > 0:
        rng = np.random.default_rng(args.seed + 1)
        y = y * (1.0 + C.Y_NOISE_FRAC * rng.standard_normal(y.shape))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, theta=theta.astype(np.float32),
             y=y.astype(np.float32),
             param_names=np.array(C.PARAM_NAMES))
    print(f"Saved {len(theta)} pairs -> {args.out}")
    print("theta ranges:")
    for j, name in enumerate(C.PARAM_NAMES):
        print(f"  {name}: [{theta[:, j].min():.4f}, {theta[:, j].max():.4f}]")
    print("y (Hz) ranges:")
    for j in range(C.Y_DIM):
        print(f"  f{j + 1}: [{y[:, j].min():.3f}, {y[:, j].max():.3f}]")


if __name__ == "__main__":
    main()
