"""
sample.py -- draw posterior samples theta ~ p(theta | y_obs) from a
trained model and report summary statistics in physical units.

Usage:
    python sample.py --method cfm --y 28.1 55.3 89.7 120.4 150.2 --n 5000
    python sample.py --method all --y 28.1 55.3 89.7 120.4 150.2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import config as C
from data import Standardiser
from models import REGISTRY


def load(method: str, ckpt_dir: Path, device):
    ck = torch.load(ckpt_dir / f"{method}.pt", map_location=device,
                    weights_only=False)
    model = REGISTRY[method](C.THETA_DIM, C.Y_DIM,
                             ck["common_cfg"], ck["method_cfg"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, Standardiser.from_state(ck["s_theta"]), \
        Standardiser.from_state(ck["s_y"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=[*REGISTRY.keys(), "all"])
    ap.add_argument("--y", type=float, nargs=C.Y_DIM, required=True,
                    help="observed first-5 natural frequencies [Hz]")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints"))
    ap.add_argument("--out", type=Path, default=Path("posterior"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    methods = list(REGISTRY) if args.method == "all" else [args.method]
    args.out.mkdir(parents=True, exist_ok=True)

    for m in methods:
        model, s_theta, s_y = load(m, args.ckpt, device)
        y = torch.from_numpy(
            s_y.transform(np.asarray([args.y], dtype=np.float32))
            .astype(np.float32)).to(device)
        with torch.no_grad():
            th = model.sample(y, args.n)          # [n, 1, D]
        th = s_theta.inverse(th[:, 0, :].cpu().numpy())
        np.save(args.out / f"samples_{m}.npy", th)

        print(f"\n[{m}] posterior from {args.n} samples "
              f"(y_obs = {args.y}):")
        for j, name in enumerate(C.PARAM_NAMES):
            q05, q50, q95 = np.percentile(th[:, j], [5, 50, 95])
            print(f"  {name:>3}: mean={th[:, j].mean():9.4f}  "
                  f"std={th[:, j].std():8.4f}  "
                  f"[5%, 50%, 95%] = [{q05:9.4f}, {q50:9.4f}, {q95:9.4f}]")


if __name__ == "__main__":
    main()
