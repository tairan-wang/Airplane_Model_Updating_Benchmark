"""
train.py -- unified trainer for all five conditional generative models.

Usage:
    python train.py --method cfm   --data data/dataset.npz
    python train.py --method cddpm --data data/dataset.npz --epochs 400
    python train.py --method all   --data data/dataset.npz

Checkpoints (weights + both Standardisers + configs) are written to
checkpoints/<method>.pt; best-validation weights are kept for likelihood/
regression-style losses, last weights for the adversarial cGAN.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch

import config as C
from data import load_dataset, loaders
from models import REGISTRY


def get_device():
    want = C.COMMON.get("device", "cpu")
    return torch.device(want if (want == "cpu" or torch.cuda.is_available())
                        else "cpu")


def train_standard(model, cfg, tr, va, device):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    best, best_state = float("inf"), None
    history = {"train": [], "val": []}
    for ep in range(cfg["epochs"]):
        model.train()
        tl, nb = 0.0, 0
        for theta, y in tr:
            theta, y = theta.to(device), y.to(device)
            out = model.loss(theta, y)
            opt.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tl += out["loss"].item(); nb += 1
        model.eval()
        with torch.no_grad():
            vl = np.mean([model.loss(t.to(device), y.to(device))
                          ["loss"].item() for t, y in va])
        history["train"].append(tl / nb)
        history["val"].append(float(vl))
        if vl < best:
            best, best_state = vl, copy.deepcopy(model.state_dict())
        if (ep + 1) % max(1, cfg["epochs"] // 10) == 0:
            print(f"  epoch {ep + 1:4d}/{cfg['epochs']}  val={vl:.4f}  "
                  f"best={best:.4f}")
    model.load_state_dict(best_state)
    return {"best_val": best}, history


def train_gan(model, cfg, tr, va, device):
    opt_g = torch.optim.Adam(model.gen_parameters(), lr=cfg["lr_g"],
                             betas=cfg["betas"])
    opt_d = torch.optim.Adam(model.disc_parameters(), lr=cfg["lr_d"],
                             betas=cfg["betas"])
    history = {"G": [], "D": []}
    for ep in range(cfg["epochs"]):
        model.train()
        model.noise_scale = cfg["instance_noise"] * max(
            0.0, 1.0 - ep / (0.8 * cfg["epochs"]))       # linear decay
        gl = dl = n = 0
        for theta, y in tr:
            theta, y = theta.to(device), y.to(device)
            d_loss = model.disc_loss(theta, y)
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()
            g_loss = model.gen_loss(theta, y)
            opt_g.zero_grad(); g_loss.backward(); opt_g.step()
            gl += g_loss.item(); dl += d_loss.item(); n += 1
        history["G"].append(gl / n)
        history["D"].append(dl / n)
        if (ep + 1) % max(1, cfg["epochs"] // 10) == 0:
            print(f"  epoch {ep + 1:4d}/{cfg['epochs']}  "
                  f"G={gl / n:.4f}  D={dl / n:.4f}  "
                  f"noise={model.noise_scale:.3f}")
    return {"final_G": gl / n, "final_D": dl / n}, history


def plot_history(method: str, history: dict, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, v in history.items():
        ax.plot(range(1, len(v) + 1), v, label=k)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(f"{method}: training history")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{method}_loss.png", dpi=150)
    plt.close(fig)


def plot_r2(method: str, model, val, s_theta, device, out_dir: Path,
            n_samp: int = 128, max_val: int = 500):
    """Posterior-mean recovery on the validation set: theta_hat vs theta,
    with R^2 per parameter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theta_v, y_v = val
    if len(theta_v) > max_val:
        theta_v, y_v = theta_v[:max_val], y_v[:max_val]
    with torch.no_grad():
        th = model.sample(y_v.to(device), n_samp)      # [n, B, D] (std.)
    mean_std = th.mean(dim=0).cpu().numpy()
    pred = s_theta.inverse(mean_std)
    true = s_theta.inverse(theta_v.numpy())

    r2s = []
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for j, ax in enumerate(axes.flat):
        t, p = true[:, j], pred[:, j]
        ss_res = float(((t - p) ** 2).sum())
        ss_tot = float(((t - t.mean()) ** 2).sum()) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        r2s.append(r2)
        ax.scatter(t, p, s=6, alpha=0.4, color="#4878a8")
        lim = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel(f"true {C.PARAM_NAMES[j]}")
        ax.set_ylabel(f"posterior mean {C.PARAM_NAMES[j]}")
        ax.set_title(f"{C.PARAM_NAMES[j]}:  R$^2$ = {r2:.3f}")
    fig.suptitle(f"{method}: posterior-mean recovery (validation)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / f"{method}_r2.png", dpi=150)
    plt.close(fig)
    return r2s


def run(method: str, data_path: Path, epochs_override, out_dir: Path):
    torch.manual_seed(C.COMMON["seed"])
    np.random.seed(C.COMMON["seed"])
    device = get_device()

    cfg = dict(C.METHOD_CONFIGS[method])
    if epochs_override:
        cfg["epochs"] = epochs_override

    train, val, s_theta, s_y = load_dataset(data_path,
                                            seed=C.COMMON["seed"])
    tr, va = loaders(train, val, cfg["batch_size"])

    model = REGISTRY[method](C.THETA_DIM, C.Y_DIM, C.COMMON, cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[{method}] device={device}  params={n_par:,}  "
          f"train={len(train[0])}  val={len(val[0])}")

    t0 = time.time()
    if getattr(model, "is_adversarial", False):
        stats, history = train_gan(model, cfg, tr, va, device)
    else:
        stats, history = train_standard(model, cfg, tr, va, device)
    stats["train_time_s"] = time.time() - t0

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    plot_history(method, history, plots_dir)
    r2s = plot_r2(method, model, val, s_theta, device, plots_dir)
    stats["val_R2"] = {n: round(r, 4)
                       for n, r in zip(C.PARAM_NAMES, r2s)}
    print(f"[{method}] validation R2: {stats['val_R2']}")

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"method": method,
                "state_dict": model.state_dict(),
                "common_cfg": C.COMMON, "method_cfg": cfg,
                "s_theta": s_theta.state_dict(),
                "s_y": s_y.state_dict(),
                "stats": stats, "history": history},
               out_dir / f"{method}.pt")
    print(f"[{method}] saved -> {out_dir / (method + '.pt')}  "
          f"({stats})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=[*REGISTRY.keys(), "all"])
    ap.add_argument("--data", type=Path, default=Path("data/dataset.npz"))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("checkpoints"))
    args = ap.parse_args()

    methods = list(REGISTRY) if args.method == "all" else [args.method]
    for m in methods:
        run(m, args.data, args.epochs, args.out)


if __name__ == "__main__":
    main()
