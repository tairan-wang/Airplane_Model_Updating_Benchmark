# Workflow: Data-Driven Model Updating of the Airplane Toy Model

**From parametric FE simulation, via GP surrogate modelling, to conditional
deep generative model updating**

Benchmark: lab-scale airplane modal test (Bi, Beer, Cogan & Mottershead,
2023, *MSSS*). Working directory: `D:\python_run_package`. All lengths in
mm, moduli in MPa (FE stage) / ×10¹¹ Pa (surrogate & generative stages),
frequencies in Hz.

---

## 0. Problem definition

| Quantity | Symbol | Dim | Description |
|---|---|---|---|
| Parameters | θ = (a, b, E1, E2) | 4 | half-wingspan position, wingtip chord, Young's moduli of the two joint-region materials |
| Observations | y = (f₁ … f₅) | 5 | first five natural frequencies |

Goal: learn the inverse (posterior) map **p(θ | y)** with five conditional
generative architectures sharing one framework, enabling a like-for-like
methodological comparison for FE model updating.

Known difficulty: the original parameterisation poorly recovers variance of
torsion-dominated modes (5th, 7th) — rank-deficient torsional stiffness
sensitivity. A contributing model-form factor identified during this work:
the 1.2 mm wing plate is meshed with **three layers of C3D8R** linear
reduced-integration hexes, a configuration known to represent thin-plate
torsion poorly.

---

## 1. Parametric FE automation (Abaqus)

### 1.1 Model archaeology

The CAE journal (`case_test-2021-2021.jnl`, ~19,600 lines) was mined to
reconstruct the manual parametric workflow previously used:

- Model `Model-3Dtest`, wing part `Avion_Metal_V6-3`, feature
  `Solid extrude-2`.
- Geometry varied by deleting and redrawing the wing outline in the feature
  sketch: unswept edge (24, 70)→(a, 70); tip chord (a, 70)→(a, 70−b);
  swept edge back to (24, −70); mirrored about the sketch centreline.
  Baseline: a = 296.75, b = 26.01.
- Materials `Al4lianjiequyu1` / `Al4lianjiequyu2` (joint regions) carry
  E1 / E2, ν = 0.33.
- Manual runs `Job-surrogate_2_N` followed exactly this recipe.

### 1.2 Two-layer automation

| Layer | File | Environment | Role |
|---|---|---|---|
| Inner | `abaqus_run_single.py` | Abaqus/CAE batch (noGUI) | one sample: edit sketch, set E1/E2, remesh, frequency step, extract results to JSON |
| Outer | `driver.py` | normal Python 3 | Sobol' sampling of θ, one isolated folder per run, subprocess management, dataset assembly |

Call chain:
`python driver.py --n N` → per sample: copy master `.cae` + script into
`runs\run_XXXX\` → `abaqus cae noGUI=... -- id a b E1 E2` →
`result_XXXX.json` → merged `runs\dataset.csv`.

### 1.3 Robustness principles (and the failures that motivated them)

| Principle | Failure it fixes |
|---|---|
| Master `.cae` copied per-run, opened locally, **never saved** | file-lock crash: OneDrive sync + open GUI session held the master (`utl_File: CreateFile` error) |
| Coordinate-based selection of wing-outline lines (any LINE with \|x\| > root plane), not repository indices | index drift makes journal-recorded deletes unreproducible |
| In-memory `part.Unlock()` + assembly unlock after opening | Abaqus 2021 model opened in CAE 2024 arrives with parts **and** assembly locked (version-migration lock) |
| Section regions recorded from the pristine model (cell bounding boxes), rebuilt after regeneration (joint by bbox, wing by boolean complement) | geometry replacement kills picked cell sets → 16,635 elements without properties → solver rejects input |
| Empty eigenmode list ⇒ `status: failed` | ODB can exist even when the input processor errored (false "ok") |
| Nearest-node sensor lookup by coordinates; assembly bbox reported in JSON | remeshing renumbers nodes; instance is translated/rotated (≈(130, −260, 382), 180° about x) so sensor coords must be global-frame |
| Distorted-element count per run | extreme (a, b) degrade the mesh (519 warnings observed in one historical run) |
| Abaqus-Python quirks respected (no generator args to `sum`) | `from abaqus import *` shadows builtins |

Operational rule: **no OneDrive** anywhere in the batch path — sync locks
scratch files (`.simdir` cleanup denied) and the master database.

### 1.4 Verification protocol

Single run at baseline θ = (296.75, 26.0087, E nominal) must return
`status: "ok"`, ten frequencies matching a historical baseline job, small
`n_distorted_elements`, and sensor-node distances of a few mm once
`SENSOR_COORDS` are set from the reported assembly bbox. Geometry edit was
validated by the regenerated span: part bbox x-extent 593.5 mm = 2 × 296.75.

---

## 2. Surrogate modelling (GP)

- **Design**: Latin hypercube over the 4-D prior box (LHS is well suited to
  small, expensive FE designs: perfect 1-D stratification).
- **Model**: multi-output Gaussian process, independent GPs per output
  (scikit-learn `MultiOutputRegressor`-style), 7 natural frequencies;
  standardised inputs via a stored scaler.
- **Units**: a, b in mm; E1, E2 in ×10¹¹ Pa.
- **Artefacts**: `surrogate\multioutput_gp.joblib` (bundle with model and
  `n_modes`), `surrogate\input_scaler.joblib`.
- **Interface**: `predict_gp.py` — point or CSV batch prediction, optional
  predictive std per output (`--return-std`), e.g.
  `python surrogate\predict_gp.py --a 300 --b 25 --E1 0.6 --E2 0.7`.

Role in the pipeline: replaces the ~minutes-per-run FE solve with a
~milliseconds evaluation, making 20k-sample generative training sets cheap.

---

## 3. Generative training data (GP → dataset)

`model_updating\generate_dataset.py`:

1. Sample θ from the prior box (`config.PRIOR_BOUNDS`) — scrambled
   **Sobol'** (`--sobol`) preferred over pseudo-random: better joint-space
   uniformity in 4-D and extensible (nested) if more samples are needed
   later. Power-of-2 n (16,384 / 32,768) gives the cleanest balance.
2. Predict 7 frequencies with the GP; keep the **first 5** (Y_DIM).
3. Add proportional Gaussian noise `Y_NOISE_FRAC` (default 0.2%) emulating
   measurement uncertainty — prevents degenerate, over-confident
   conditionals; should be tuned to the actual modal-test noise level.
4. Save `data\dataset.npz` (θ, y, names); ranges printed for sanity checks.

Command:
```
python generate_dataset.py --n 20000 --gp-dir ..\surrogate --sobol
```

Design-of-experiments summary: **LHS for the small FE→GP stage, Sobol' for
the large GP→generative stage** — each method where its strengths matter.

---

## 4. Conditional generative framework (five methods, one interface)

Package `model_updating\` (PyTorch). Shared components in
`models\common.py`:

- **ConditionEmbedder** — one MLP y ∈ R⁵ → context h ∈ R⁶⁴, identical for
  all methods (the "conditional network" of the framework).
- **TimeEmbedding** — sinusoidal + MLP, used by the two dynamical methods.
- **Base interface** — every model implements `loss(θ, y)` and
  `sample(y, n)`; θ and y are z-scored internally, scalers stored in every
  checkpoint.

Method-specific networks:

| Method | Generative mechanism | Training objective | Sampling cost |
|---|---|---|---|
| cVAE | latent z (6-D), Gaussian decoder with learned variance | β-weighted ELBO | 1 pass |
| cGAN | G(z, h) vs D(θ, h), decaying instance noise | non-saturating BCE | 1 pass |
| cNF | 8 affine coupling layers (RealNVP), tanh-clamped scales | exact NLL | 1 inverse pass |
| cDDPM | ε-prediction, linear β schedule, T = 300 | denoising MSE | 300 passes |
| cFM | velocity field on linear (σ_min) interpolant | flow-matching MSE | 100 Euler steps (tunable) |

Notes: cNF provides exact `log_prob` (quantitative reference); cFM uses
independent coupling with minibatch **OT-CFM** as a documented drop-in
extension; cGAN is the likelihood-free baseline and the least stable.

---

## 5. Training and posterior sampling

```
python train.py --method all --data data\dataset.npz
python sample.py --method all --y f1 f2 f3 f4 f5 --n 5000
```

- Unified trainer: AdamW, gradient clipping, 90/10 train/val split,
  best-validation checkpointing (last-epoch for the adversarial cGAN);
  GPU used automatically if present.
- Checkpoints `checkpoints\<method>.pt` are self-contained (weights +
  scalers + configs).
- `sample.py` maps the observed frequencies through the stored y-scaler,
  draws n posterior samples, restores physical units, prints
  mean/std/5-50-95% quantiles per parameter, and saves
  `posterior\samples_<method>.npy` for corner plots.
- Framework is dimension-agnostic: adding MAC/mode-shape features to break
  the torsional rank-deficiency only changes `Y_DIM` and dataset
  generation.

Whole-framework smoke test passed: all five methods trained and sampled
end-to-end on a synthetic θ→y map before deployment.

---

## 6. Validation & next steps

- **Statistical validation** (uniform across methods via the shared
  `sample()` interface): simulation-based calibration (SBC) / rank
  histograms, TARP coverage, posterior predictive checks by pushing θ
  samples back through the GP and comparing predicted vs observed f₁–f₅.
- **Physics validation**: FE re-analysis at posterior mean/MAP θ;
  MAC against measured mode shapes.
- **Expected structure**: wide/correlated posteriors along weakly
  identified directions reflecting the f5/f7 torsional identifiability
  problem — consistency of this structure across the five methods (with
  cNF's exact likelihood as reference) is itself a key comparison result.
- **Open directions**: OT-CFM coupling; mode-shape (MAC) features in y;
  directional joint-stiffness parameterisation; shell-element wing variant
  to quantify the C3D8R model-form error.

---

## Appendix A. File map

```
D:\python_run_package\
├── config_dataset.yaml          # FE campaign config (workers, cpus, memory, timeouts)
├── driver.py                    # FE campaign (subcommands: sample|generate-inp|
│                                #   solve-parallel|extract|build|all)
├── abaqus_run_single.py         # per-sample Abaqus/CAE batch script (write_inp)
├── extract_odb.py               # per-run ODB -> result_XXXX.json (Abaqus python)
├── build_dataset.py             # aggregate result JSONs -> dataset\train.npz
├── rebuild_from_new_fem.py      # one-shot orchestrator (clean|fe|train-gp|
│                                #   gen-dataset|train-gen)
├── mac_extract.py               # full-field auto/cross MAC (Abaqus python)
├── mode_symmetry.py             # sym/antisym wing-mode classifier (Abaqus python)
├── samples\
│   ├── lhs_200_seed42.csv       # LHS design
│   ├── runs\run_XXXX\           # per-run scratch (inp, odb, result_XXXX.json)
│   └── solve_progress.log       # live solve status
├── dataset\train.npz            # FE training set (theta, 7 freqs)
├── surrogate\                   # multioutput_gp.joblib, input_scaler.joblib,
│                                #   train_gp.py, predict_gp.py, metrics/plots
└── model_updating\
    ├── config.py                # dims, prior box, hyperparameters, A_OFFSET(=24)
    ├── generate_dataset.py      # GP -> (theta, y) dataset (Sobol')
    ├── data.py                  # standardisation, loaders
    ├── models\{common,cvae,cgan,cnf,cddpm,cfm}.py
    ├── train.py                 # unified trainer (--method all)
    ├── sample.py                # single-obs posterior sampling (a shifted +24)
    ├── infer_obs.py             # batch posterior over all obs -> results\ + plots
    ├── validate_surrogate_f7.py # held-out validation: surrogate f7 vs obs f6
    ├── validation_hist_pdf.py           # histogram + fitted KDE (f7 vs f6)
    ├── validation_hist_all_modes.py     # all-mode distributional grid
    ├── data\obs_natural_frequencies_10.csv  # observed f1..f10 (only f1..f5 used)
    ├── data\obs_input.csv       # logged truth (a, b) for the 30 structures
    └── results\                 # samples_*.csv, pairplots, recovery, validation, MAC
```

## Appendix B. Command runbook (end-to-end)

> **Run everything from PowerShell** in `D:\python_run_package` (Abaqus must be
> on PATH). Keep the whole path **off OneDrive**. All stages **resume** — they
> skip completed runs — so any command below is safe to re-run after a stop.
> If a Python command hangs with **no output at all**, the Windows **WMI
> service is stuck** — reboot (see Force-stop).

### Key config (`config_dataset.yaml`)

```yaml
n_samples: 200        seed: 42        n_modes_keep: 7
gen_inp_workers: 1    # MUST stay 1 (parallel CAE clobbers sibling jobs)
max_workers: 2        # concurrent solves   (2 x 4 cpus = 8 threads)
cpus_per_job: 4
memory_per_job: "4 gb" -> set "24 gb"   # ~1M-DOF model needs in-core RAM
timeout_solve_s: 5400 # 90 min (a single solve is ~30 min)
```

### Stage 1 — FE campaign (Abaqus; the long stage, ~hours)

```powershell
python -u driver.py sample          # LHS design           -> samples\lhs_200_seed42.csv
python -u driver.py generate-inp    # CAE noGUI -mesa       -> run_XXXX.inp   (serial)
python -u driver.py solve-parallel  # abaqus solve (interactive, 24 GB/job)
python -u driver.py extract         # ODB -> result_XXXX.json (7 freqs)
python -u driver.py build           # aggregate            -> dataset\train.npz
python -u driver.py all             # ...or all five stages in sequence
```

### Stage 2 — GP surrogate

```powershell
python -u surrogate\train_gp.py                              # -> surrogate\*.joblib
python surrogate\predict_gp.py --a 277 --b 25 --E1 0.6 --E2 0.7   # spot check (sketch-frame a)
```

### Stage 3 — generative training set (GP -> data)

```powershell
cd model_updating
python -u generate_dataset.py --n 20000 --gp-dir ..\surrogate --seed 42 --sobol --out data\dataset.npz
```

### Stage 4 — train the five methods

```powershell
python -u train.py --method all --data data\dataset.npz     # -> checkpoints\*.pt
```

### Stage 5 — posterior generation (`a` auto-shifted +24 to physical frame)

```powershell
python -u infer_obs.py --method all --n 5000                # 30 obs -> results\samples_*.csv + pair/recovery plots
python -u sample.py --method all --y 19.6 39.8 88.7 103.1 141.2 --n 5000   # single observation
```

### Stage 6 — validation

```powershell
python -u validate_surrogate_f7.py        # held-out: surrogate f7 vs observed f6 (recovery)
python -u validation_hist_pdf.py          # histogram + fitted KDE (f7 vs f6)
python -u validation_hist_all_modes.py    # all-mode distributional grid
```

### Stage 7 — MAC / mode analysis (Abaqus python; reads existing ODBs)

```powershell
cd D:\python_run_package
abaqus python mac_extract.py -- 100       # full-field auto-MAC -> results\mac_run0100.npz
abaqus python mode_symmetry.py -- 100     # symmetric / antisymmetric wing-mode table
```

### One-shot orchestrator (`rebuild_from_new_fem.py`)

```powershell
python -u rebuild_from_new_fem.py --n 200 --seed 42 --max-workers 2 --n-gen 20000   # FULL (DELETES samples\runs first!)
python -u rebuild_from_new_fem.py --from-stage train-gp --n-gen 20000 --seed 42     # resume from GP (keeps FE data)
python -u rebuild_from_new_fem.py --from-stage fe                                    # resume FE, skip the destructive clean
```

---

## Appendix C. Force-stop, recovery & status

### Stop a running job

```powershell
# 1. Graceful: press Ctrl+C in the driver's terminal.

# 2. Hard stop (Ctrl+C did not reach the Abaqus children) -- run as ADMINISTRATOR:
Get-Process standard, pre, explicit, ABQcaeK, ABQcaeG, SMAPython, SMALauncher, SMAEqsDirSolverSymmetric -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like '*Python312*' } | Stop-Process -Force

# 3. Clear stale lock files before restarting:
Get-ChildItem samples\runs -Recurse -Filter *.lck -ErrorAction SilentlyContinue | Remove-Item -Force
```

Then just re-run the stage — it resumes from where it stopped.

### A solver that will not die

If `taskkill /F /PID <id>` returns **"timeout period expired"** and the
`standard.exe` keeps accumulating CPU, it is wedged in an uninterruptible
kernel I/O wait and **cannot be killed** — **reboot**. A reboot also clears a
hung WMI service (the cause of a driver that starts but prints nothing).

### Recover individual failed runs

```powershell
# Corrupt CAE copy  (result: "openMdb ... invalid model database", no .inp):
Remove-Item samples\runs\run_0034\master_model.cae, samples\runs\run_0034\run_0034.* -Force
python -u driver.py generate-inp; python -u driver.py solve-parallel; python -u driver.py extract

# Partial / killed solve  (ODB exists but "no non-zero eigenmodes" at extract):
Remove-Item samples\runs\run_0027\run_0027.odb, samples\runs\run_0027\run_0027.lck -Force
python -u driver.py solve-parallel; python -u driver.py extract
```

### Check progress

```powershell
Get-Content samples\solve_progress.log -Tail 20          # live solve log
(Get-ChildItem samples\runs\*\*.odb).Count               # completed solves
Get-Process standard -ErrorAction SilentlyContinue       # is a solver running now?
```
