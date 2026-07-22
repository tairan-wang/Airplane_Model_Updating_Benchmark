"""
validate_surrogate_f7.py -- posterior-predictive validation on a HELD-OUT
high-order natural frequency, with correct FE<->test mode pairing.

Inference used only f1..f5, so higher modes are never seen during inference.
The naive check (surrogate f7 vs observed f7) is misleading because the FE
model carries a near-degenerate wing doublet at ~142 Hz (f5 == f6) that the
measurement resolves as a SINGLE mode. That shifts the mode numbering by one
above ~140 Hz, so the physically matched pair is:

    surrogate f7  (0-based index 6)   <->   observed f6  (column f6_Hz)

For each method we take the posterior samples (a, b, E1, E2), convert 'a'
back to the FE sketch frame (- A_OFFSET, since the surrogate trained there),
push them through the multi-output GP surrogate, read the predicted surrogate
f7, and compare its per-observation posterior-predictive distribution against
the observed f6.
"""
from __future__ import annotations

import csv
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as C

HERE = Path(__file__).resolve().parent
SUR = HERE.parent / "surrogate"
RES = HERE / "results"
OBS_CSV = HERE / "data" / "obs_natural_frequencies_10.csv"
METHODS = ["cddpm", "cfm", "cgan", "cnf", "cvae"]

# Physically matched pair (see module docstring): surrogate mode 7 vs test f6.
SUR_MODE = 7                    # surrogate mode number (1-based)
SUR_IDX = SUR_MODE - 1          # -> 0-based index into predict() output
OBS_MODE = 6                    # observed mode number (1-based)
OBS_COL = f"f{OBS_MODE}_Hz"     # column name in the observation CSV


def load_surrogate():
    bundle = joblib.load(SUR / "multioutput_gp.joblib")
    scaler = joblib.load(SUR / "input_scaler.joblib")
    return bundle["model"], scaler, int(bundle.get("n_modes", 7))


def read_obs_freq(col_name):
    with OBS_CSV.open() as f:
        r = csv.reader(f)
        header = next(r)
        col = header.index(col_name)
        return np.array([float(row[col]) for row in r if row])


def load_posterior(method):
    obs, th = [], []
    with (RES / f"samples_{method}.csv").open(newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            obs.append(int(float(row[0])))
            th.append([float(x) for x in row[1:5]])   # a,b,E1,E2 (a physical)
    return np.asarray(obs), np.asarray(th, dtype=np.float64)


def main():
    model, scaler, n_modes = load_surrogate()
    obs_freq = read_obs_freq(OBS_COL)
    print(f"surrogate n_modes={n_modes}, observations={len(obs_freq)}")
    print(f"comparing SURROGATE f{SUR_MODE} (idx {SUR_IDX})  vs  "
          f"OBSERVED f{OBS_MODE} ({OBS_COL})")

    fig, axes = plt.subplots(1, len(METHODS),
                             figsize=(4.0 * len(METHODS), 4.2), sharey=True)
    stats = {}
    for ax, m in zip(axes, METHODS):
        obs_id, th = load_posterior(m)
        th_sur = th.copy()
        th_sur[:, 0] -= C.A_OFFSET               # physical 'a' -> sketch frame
        Xs = scaler.transform(th_sur)
        fpred = model.predict(Xs)[:, SUR_IDX]    # [N] predicted surrogate f7

        ids = np.unique(obs_id)
        mean = np.array([fpred[obs_id == i].mean() for i in ids])
        lo = np.array([np.percentile(fpred[obs_id == i], 5) for i in ids])
        hi = np.array([np.percentile(fpred[obs_id == i], 95) for i in ids])
        o = obs_freq[ids]

        cov = float(np.mean((lo <= o) & (o <= hi)))
        rmse = float(np.sqrt(np.mean((mean - o) ** 2)))
        bias = float(np.mean(mean - o))
        stats[m] = (cov, rmse, bias)

        ax.errorbar(o, mean, yerr=[mean - lo, hi - mean], fmt="o", ms=4,
                    lw=0.8, capsize=2, color="#4878a8", ecolor="#9db8d2",
                    label="posterior predictive (5-95%)")
        lim = [min(o.min(), lo.min()), max(o.max(), hi.max())]
        ax.plot(lim, lim, "k--", lw=1, label="y = x")
        ax.set_xlabel(f"observed $f_{OBS_MODE}$ [Hz]")
        ax.set_title(f"{m}\n90% CI cov={cov:.2f} | RMSE={rmse:.1f} Hz")
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel(f"surrogate-predicted $f_{SUR_MODE}$ [Hz]\n"
                       "(posterior mean, 5-95%)")
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Validation: posterior -> GP surrogate $f_{SUR_MODE}$  vs  "
                 f"observed $f_{OBS_MODE}$  (matched across the ~142 Hz FE "
                 f"doublet)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = RES / f"validation_surf{SUR_MODE}_vs_obsf{OBS_MODE}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    print("\n method   90%cov   RMSE[Hz]   bias[Hz]")
    for m in METHODS:
        c, r, b = stats[m]
        print(f"  {m:6s}  {c:5.2f}   {r:7.2f}   {b:+7.2f}")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
