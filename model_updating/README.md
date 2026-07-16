# model_updating — conditional generative models for FE model updating

Five deep generative methods learn the inverse map p(θ | y) for the
airplane toy model: condition **y** = first five natural frequencies (Hz),
generated sample **θ** = (a [mm], b [mm], E1, E2 [×1e11 Pa]).

All methods share the same **ConditionEmbedder** (y → context h) defined in
`models/common.py`; only the generative mechanism differs:

| method | file | mechanism | training loss | sampling |
|---|---|---|---|---|
| cVAE | `models/cvae.py` | latent z + Gaussian decoder | ELBO (β-VAE) | z ~ N(0,I) → decoder |
| cGAN | `models/cgan.py` | generator vs discriminator | non-saturating BCE + instance noise | G(z, h) |
| cNF | `models/cnf.py` | affine-coupling flow (RealNVP) | exact NLL | inverse flow |
| cDDPM | `models/cddpm.py` | ε-prediction diffusion (T=300) | denoising MSE | ancestral, T steps |
| cFM | `models/cfm.py` | velocity field on linear/OT path | flow-matching MSE | Euler ODE, 100 steps |

## Workflow

```bat
cd D:\python_run_package\model_updating

:: 1. generate training data from the trained GP surrogate
python generate_dataset.py --n 20000 --gp-dir ..\surrogate --sobol

:: 2. train (any single method, or all five)
python train.py --method cfm --data data\dataset.npz
python train.py --method all --data data\dataset.npz

:: 3. posterior sampling for an observed frequency vector
python sample.py --method all --y 28.1 55.3 89.7 120.4 150.2 --n 5000
```

Requires: `torch`, `numpy`, `scipy`, `joblib` (same env as the GP).
Checkpoints in `checkpoints/<method>.pt` include the weights, both
standardisers, and all hyperparameters — `sample.py` is self-contained.

## Configuration

Everything lives in `config.py`: prior bounds (keep consistent with the GP
training range and units — E in 1e11 Pa as in `predict_gp.py`), the shared
embedder size, and per-method hyperparameters. `Y_NOISE_FRAC` adds
proportional Gaussian noise to the GP frequencies during dataset
generation, emulating measurement noise (0.2% default).

## Notes for the method comparison

- **cNF** gives exact conditional log-likelihoods (`CNF.log_prob`) — useful
  for quantitative comparison and for validation diagnostics.
- **cFM** uses independent (x0, x1) coupling; minibatch **OT-CFM** is a
  drop-in extension (reorder x0 by an OT assignment inside `loss`).
- **cGAN** has no likelihood and is the least stable — treat as baseline.
- Sampling cost differs sharply: cVAE/cGAN/cNF are single-pass; cDDPM needs
  T=300 network evaluations, cFM 100 (tunable via `ode_steps`).
- Suggested validation for the paper: SBC / rank histograms and posterior
  predictive checks by pushing θ samples back through the GP; both operate
  on the shared `sample()` interface, so one evaluation script covers all
  five methods.

## Extending to mode-shape features

When MAC-based features are added to break the f5/f7 torsional
rank-deficiency, only `Y_DIM` in `config.py` and the dataset generation
change — the framework is dimension-agnostic.
