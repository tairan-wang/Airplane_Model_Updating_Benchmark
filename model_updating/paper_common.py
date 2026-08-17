"""
paper_common.py -- shared helpers for the paper re-analysis (results/paper_1000/).

Read-only w.r.t. the existing pipeline: reuses the trained GP + checkpoints and
config.A_OFFSET/to_physical_frame. Nothing here retrains or modifies the 24 mm
offset logic. All paper outputs live under results/paper_1000/.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import config as C

# numpy 2.x renamed trapz -> trapezoid; support both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "results" / "paper_1000"
DATA = HERE / "data"
CKPT = HERE / "checkpoints"
SUR = ROOT / "surrogate"

OBS_FREQ_CSV = DATA / "obs_natural_frequencies_10.csv"
OBS_INPUT_CSV = DATA / "obs_input.csv"

METHODS = ["cvae", "cgan", "cnf", "cddpm", "cfm"]
METHOD_LABEL = {"cvae": "cVAE", "cgan": "cGAN", "cnf": "cNF",
                "cddpm": "cDDPM", "cfm": "cFM"}

# Physical display/compute ranges (a in physical frame; +24 stays internal).
A_RANGE = (290.0, 310.0)
B_RANGE = (20.0, 30.0)
E_RANGE = (0.50, 0.90)
PARAM_RANGE = {"a": A_RANGE, "b": B_RANGE, "E1": E_RANGE, "E2": E_RANGE}

# Paper mode display: Mode 1..6 = surrogate output indices below (f6 hidden).
MODE_SURR_IDX = [0, 1, 2, 3, 4, 6]        # surrogate f1..f5, f7
MODE_LABELS = ["Mode 1", "Mode 2", "Mode 3", "Mode 4", "Mode 5", "Mode 6"]
# Held-out "sixth modal frequency": surrogate f7 (idx 6) vs observed f6_Hz.
HELDOUT_SURR_IDX = 6
HELDOUT_OBS_COL = "f6_Hz"
# In-domain conditioning modes f1..f5 -> surrogate indices 0..4.
INDOMAIN_SURR_IDX = [0, 1, 2, 3, 4]
INDOMAIN_OBS_COLS = ["f1_Hz", "f2_Hz", "f3_Hz", "f4_Hz", "f5_Hz"]

# Okabe-Ito colour-blind-safe palette (5 methods + reference).
METHOD_COLOR = {
    "cvae": "#0072B2", "cgan": "#E69F00", "cnf": "#009E73",
    "cddpm": "#CC79A7", "cfm": "#56B4E9",
}
REF_COLOR = "#000000"


# ----------------------------------------------------------------------
# Readers (obs frequencies + logged truth)
# ----------------------------------------------------------------------
def read_obs_cond():
    """Conditioning observations y = f1..f5 [N_obs, 5]."""
    rows = list(csv.DictReader(OBS_FREQ_CSV.open(encoding="utf-8")))
    return np.array([[float(r[c]) for c in INDOMAIN_OBS_COLS] for r in rows],
                    dtype=np.float64)


def read_obs_col(col):
    """One observed frequency column, e.g. 'f6_Hz' -> [N_obs]."""
    rows = list(csv.DictReader(OBS_FREQ_CSV.open(encoding="utf-8")))
    return np.array([float(r[col]) for r in rows], dtype=np.float64)


def read_truth_ab():
    """Logged truth (a, b) [N_obs, 2] from obs_input.csv (physical a)."""
    out = []
    for line in OBS_INPUT_CSV.open(encoding="utf-8"):
        vals = []
        for p in line.replace(";", ",").split(","):
            try:
                vals.append(float(p.strip()))
            except ValueError:
                pass
        if vals:
            out.append(vals[:2])
    return np.asarray(out, dtype=np.float64)


def load_samples_csv(method, out_dir=OUT):
    """Return (obs_id[N], theta[N,4]) from paper_1000/samples_<method>.csv."""
    obs, th = [], []
    with (out_dir / f"samples_{method}.csv").open(newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            obs.append(int(float(row[0])))
            th.append([float(x) for x in row[1:5]])
    return np.asarray(obs), np.asarray(th, dtype=np.float64)


# ----------------------------------------------------------------------
# Surrogate (subtracts A_OFFSET so the GP sees sketch-frame 'a')
# ----------------------------------------------------------------------
def load_gp():
    import joblib
    bundle = joblib.load(SUR / "multioutput_gp.joblib")
    scaler = joblib.load(SUR / "input_scaler.joblib")
    return bundle["model"], scaler, int(bundle.get("n_modes", 7))


def gp_predict(model, scaler, theta_phys):
    """theta_phys[:,0] is physical a -> subtract A_OFFSET before the GP.
    Returns [N, n_modes] predicted frequencies."""
    X = np.asarray(theta_phys, dtype=np.float64).copy()
    X[:, 0] -= C.A_OFFSET
    return np.asarray(model.predict(scaler.transform(X)))


# ----------------------------------------------------------------------
# KDE (fixed absolute bandwidth) + Bhattacharyya distance
# ----------------------------------------------------------------------
def silverman_h(ref):
    """Silverman's rule bandwidth from the reference sample."""
    x = np.asarray(ref, dtype=np.float64)
    n = len(x)
    std = x.std(ddof=1) if n > 1 else 0.0
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    a = min(std, iqr / 1.349) if iqr > 0 else std
    if a <= 0:
        a = std if std > 0 else 1.0
    return float(0.9 * a * n ** (-1.0 / 5.0))


def fixed_kde(samples, grid, h):
    """Gaussian KDE evaluated on `grid` with a *fixed absolute* bandwidth `h`
    (kernel std = h), trapezoid-normalised to integrate to 1 on the grid."""
    from scipy.stats import gaussian_kde
    s = np.asarray(samples, dtype=np.float64)
    std = s.std(ddof=1)
    if std <= 0:
        std = 1.0
    kde = gaussian_kde(s, bw_method=h / std)      # -> kernel std = h
    d = np.asarray(kde(grid), dtype=np.float64)
    area = _trapz(d, grid)
    return d / area if area > 0 else d


def bhattacharyya(p, q, grid):
    """D_B = -log( integral sqrt(p q) dx )."""
    bc = float(_trapz(np.sqrt(np.clip(p * q, 0.0, None)), grid))
    bc = min(max(bc, 1e-12), 1.0)
    return float(-np.log(bc))


def bdist_1d(method_samples, ref_samples, lo, hi, n_grid=4096):
    """Bhattacharyya distance between a method's pooled distribution and a
    reference distribution, using ONE Silverman bandwidth (from the reference),
    the same grid and range for both. Returns (D_B, grid, p, q, h)."""
    grid = np.linspace(lo, hi, n_grid)
    h = silverman_h(ref_samples)
    p = fixed_kde(method_samples, grid, h)
    q = fixed_kde(ref_samples, grid, h)
    return bhattacharyya(p, q, grid), grid, p, q, h


# ----------------------------------------------------------------------
# Prior-predictive frequencies (sample theta from the sketch-frame prior box
# and push through the GP directly -- no A_OFFSET here, the prior box is sketch).
# ----------------------------------------------------------------------
def prior_predictive(model, scaler, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    lo = np.array([C.PRIOR_BOUNDS[k][0] for k in C.PARAM_NAMES])
    hi = np.array([C.PRIOR_BOUNDS[k][1] for k in C.PARAM_NAMES])
    theta = lo + rng.random((n, len(lo))) * (hi - lo)        # sketch frame
    return np.asarray(model.predict(scaler.transform(theta)))  # [n, n_modes]


# ----------------------------------------------------------------------
# Reference-style overlay: Measured f / Measured PDF / Updated f / Updated PDF
# / Prior PDF  (cyan hist + blue curve / pink hist + red curve / black dashed).
# ----------------------------------------------------------------------
UPD_HIST = "#F2A9CE"     # Updated f  (pink)
UPD_PDF = "#C1121F"      # Updated PDF (red)
MEA_HIST = "#8FE3F2"     # Measured f (cyan)
MEA_PDF = "#1B2FBF"      # Measured PDF (blue)
PRIOR_PDF = "#000000"    # Prior PDF (black dashed) -- only on posterior figures
# Frequency figures: NO prior (Updated/Measured only).
UMV_LEGEND = ["Updated f", "Updated PDF", "Measured f", "Measured PDF"]


def plot_umv(ax, updated, measured, xr, bins=18):
    """Updated/Measured overlay (no prior) for the FREQUENCY validation figures.
    Handles returned in UMV_LEGEND order for a shared figure legend."""
    from scipy.stats import gaussian_kde
    grid = np.linspace(xr[0], xr[1], 400)
    edges = np.linspace(xr[0], xr[1], bins + 1)
    hu = ax.hist(updated, bins=edges, density=True, color=UPD_HIST,
                 alpha=0.55, edgecolor="#B26A8E", lw=0.4)[2]
    lu, = ax.plot(grid, gaussian_kde(updated)(grid), color=UPD_PDF, lw=1.7)
    hm = ax.hist(measured, bins=edges, density=True, color=MEA_HIST,
                 alpha=0.50, edgecolor="#3FA9C0", lw=0.4)[2]
    lm, = ax.plot(grid, gaussian_kde(measured)(grid), color=MEA_PDF, lw=1.7)
    ax.set_xlim(xr)
    return [hu[0], lu, hm[0], lm]


def prior_flat(ax, prange, **kw):
    """Uniform-prior PDF for a parameter marginal: a flat black dashed line at
    1/width across the box range. Used ONLY on posterior (parameter) figures."""
    lo, hi = prange
    dens = 1.0 / (hi - lo)
    opts = dict(color=PRIOR_PDF, lw=1.2, ls="--")
    opts.update(kw)
    return ax.plot([lo, hi], [dens, dens], **opts)[0]


# ----------------------------------------------------------------------
def ensure_out():
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT
