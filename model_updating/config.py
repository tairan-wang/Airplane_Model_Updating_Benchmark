"""
config.py -- central configuration for the conditional generative
model-updating framework.

Problem definition
------------------
theta = (a [mm], b [mm], E1 [1e11 Pa], E2 [1e11 Pa])   -> dim 4
y     = (f1, ..., f5) natural frequencies [Hz]          -> dim 5

All generative models learn the posterior-style conditional p(theta | y):
condition = y, generated sample = theta.
"""

THETA_DIM = 4
Y_DIM = 5

# Prior box for theta -- keep consistent with the GP training range.
# Units: a [mm], b [mm], E1/E2 [x 1e11 Pa]  (matches predict_gp.py)
PRIOR_BOUNDS = {
    "a":  (290.0, 310.0),
    "b":  (20.0, 30.0),
    "E1": (0.50, 0.90),
    "E2": (0.50, 0.90),
}
PARAM_NAMES = ["a", "b", "E1", "E2"]

# Gaussian noise added to GP frequencies during dataset generation, as a
# FRACTION of each frequency (e.g. 0.002 = 0.2%). Emulates measurement
# noise; set to 0.0 for a noise-free dataset.
Y_NOISE_FRAC = 0.002

# ---------------------------------------------------------------------
# Shared architecture
# ---------------------------------------------------------------------
COMMON = {
    "cond_embed_dim": 64,      # output size of the shared condition network
    "cond_hidden": [128, 128], # hidden layers of the condition network
    "device": "cuda",          # falls back to cpu automatically
    "seed": 42,
}

# ---------------------------------------------------------------------
# Per-method hyperparameters
# ---------------------------------------------------------------------
CVAE = {
    "z_dim": 6,
    "enc_hidden": [128, 128],
    "dec_hidden": [128, 128],
    "beta": 1.0,               # KL weight (beta-VAE)
    "lr": 1e-3, "epochs": 500, "batch_size": 256,
}

CGAN = {
    "z_dim": 8,
    "gen_hidden": [128, 128],
    "disc_hidden": [128, 128],
    "lr_g": 2e-4, "lr_d": 2e-4, "betas": (0.5, 0.999),
    "instance_noise": 0.05,    # decayed to 0 over training; stabilises D
    "epochs": 800, "batch_size": 256,
}

CNF = {                        # conditional normalising flow (affine coupling)
    "n_coupling": 8,
    "coupling_hidden": [128, 128],
    "lr": 1e-3, "epochs": 500, "batch_size": 256,
}

CDDPM = {
    "T": 300,                  # diffusion steps
    "beta_start": 1e-4, "beta_end": 0.02,
    "net_hidden": [256, 256, 256],
    "time_embed_dim": 64,
    "lr": 1e-3, "epochs": 800, "batch_size": 256,
}

CFM = {                        # conditional flow matching (rectified/OT path)
    "sigma_min": 1e-3,
    "net_hidden": [256, 256, 256],
    "time_embed_dim": 64,
    "ode_steps": 100,          # Euler steps at sampling time
    "lr": 1e-3, "epochs": 500, "batch_size": 256,
}

METHOD_CONFIGS = {
    "cvae": CVAE, "cgan": CGAN, "cnf": CNF, "cddpm": CDDPM, "cfm": CFM,
}
