"""
paper_posterior.py -- Section I (seed-controlled posterior samples) + Section VI
posterior-generation timing. Writes only under results/paper_1000/.

Reuses the existing checkpoints and config.to_physical_frame (+24 stays internal);
no retraining. 5 methods x 30 observations x 1000 samples = 30,000 samples/method.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

import config as C
import paper_common as P
from infer_obs import load_model, per_obs_summary, write_summary

SEED = 20240722
N_SAMPLES = 1000          # per observation
TIMING_REPEATS = 3


def device_str():
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate(method, y_all, device):
    """Seed-controlled posterior for one method -> physical th_phys [n, N_obs, 4]."""
    model, s_theta, s_y = load_model(method, P.CKPT, device)
    y_std = torch.from_numpy(
        s_y.transform(y_all).astype(np.float32)).to(device)
    set_seed(SEED)
    with torch.no_grad():
        th = model.sample(y_std, N_SAMPLES)            # [n, N_obs, 4]
    th = th.cpu().numpy()
    th_phys = s_theta.inverse(
        th.reshape(-1, C.THETA_DIM)).reshape(th.shape)
    th_phys = C.to_physical_frame(th_phys)             # sketch -> physical a
    return model, s_theta, s_y, y_std, th_phys


def time_method(model, y_std):
    """Timed 30x1000 generation: 1 warm-up + TIMING_REPEATS timed runs.
    Excludes checkpoint loading / I/O / plotting. Returns (mean_s, std_s, runs)."""
    cuda = y_std.is_cuda
    with torch.no_grad():
        _ = model.sample(y_std, N_SAMPLES)             # warm-up
        if cuda:
            torch.cuda.synchronize()
        runs = []
        for _ in range(TIMING_REPEATS):
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model.sample(y_std, N_SAMPLES)
            if cuda:
                torch.cuda.synchronize()
            runs.append(time.perf_counter() - t0)
    return float(np.mean(runs)), float(np.std(runs)), runs


def main():
    out = P.ensure_out()
    device = torch.device(device_str())
    y_all = P.read_obs_cond()                           # [30, 5]
    truth = P.read_truth_ab()                           # [30, 2]
    n_obs = len(y_all)
    print(f"device={device}  n_obs={n_obs}  n_samples/obs={N_SAMPLES}")

    summary_rows = []
    timing = {}
    for m in P.METHODS:
        model, s_theta, s_y, y_std, th_phys = generate(m, y_all, device)
        pooled = th_phys.transpose(1, 0, 2).reshape(-1, C.THETA_DIM)  # [30000,4]
        obs_id = np.repeat(np.arange(n_obs), N_SAMPLES)

        # ---- CSV ----
        with (out / f"samples_{m}.csv").open("w", newline="") as f:
            f.write("obs_id,a,b,E1,E2\n")
            for k in range(len(pooled)):
                f.write("%d,%.6f,%.6f,%.6f,%.6f\n"
                        % (obs_id[k], *pooled[k]))
        # ---- NPY (theta [30000,4]; obs_id = repeat(arange(30),1000)) ----
        np.save(out / f"samples_{m}.npy", pooled.astype(np.float32))
        # ---- per-obs summary rows ----
        summary_rows.extend(per_obs_summary(m, th_phys, truth))

        # ---- timing ----
        mean_s, std_s, runs = time_method(model, y_std)
        timing[m] = {"mean_s": mean_s, "std_s": std_s, "runs_s": runs}
        print(f"[{m}] a=%.1f b=%.1f  |  gen 30x1000: %.3f +/- %.3f s"
              % (pooled[:, 0].mean(), pooled[:, 1].mean(), mean_s, std_s))

    write_summary(summary_rows, out / "summary.csv",
                  truth.shape[1] if truth is not None else 0)

    # ---- metadata ----
    meta = {
        "seed": SEED,
        "methods": P.METHODS,
        "n_obs": int(n_obs),
        "n_samples_per_obs": N_SAMPLES,
        "n_samples_per_method": int(n_obs * N_SAMPLES),
        "device": str(device),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "a_frame": "physical [290,310] (config.A_OFFSET=%.0f applied on output)"
                   % C.A_OFFSET,
        "mode_map": {P.MODE_LABELS[i]: "surrogate_f%d" % (P.MODE_SURR_IDX[i] + 1)
                     for i in range(len(P.MODE_LABELS))},
        "heldout_sixth": "surrogate_f%d vs observed %s"
                         % (P.HELDOUT_SURR_IDX + 1, P.HELDOUT_OBS_COL),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))

    # ---- timing metadata ----
    tmeta = {
        "device": str(device),
        "cpu": os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "gpu": (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else None),
        "torch": torch.__version__,
        "batch_strategy": "all 30 observations in one model.sample(y,1000) call",
        "seed": SEED,
        "n_samples_per_method": int(n_obs * N_SAMPLES),
        "warmup_runs": 1,
        "timed_repeats": TIMING_REPEATS,
        "excludes": "checkpoint loading, file I/O, plotting",
        "per_method": timing,
    }
    (out / "timing_posterior.json").write_text(json.dumps(tmeta, indent=2))
    print(f"-> {out}/  (samples, summary.csv, metadata.json, timing_posterior.json)")


if __name__ == "__main__":
    main()
