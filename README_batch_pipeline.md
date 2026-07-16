# Airplane toy model — parametric modal analysis dataset pipeline

Two-layer stack: Abaqus scripts (CAE / solver / ODB) + host Python driver for
LHS sampling and local parallel solves.

## Files

| File | Where it runs | Purpose |
|---|---|---|
| `abaqus_run_single.py` | Abaqus/CAE `noGUI` | Edit wing sketch `(a,b)`, set `E1/E2`, mesh; mode `write_inp` writes `.inp` only, mode `full` also solves + extracts |
| `extract_odb.py` | `abaqus python` | Read ODB → `result_XXXX.json` (7 non-zero frequencies + sensor modes) |
| `driver.py` | Normal Python 3 | LHS sample → generate-inp → solve-parallel → extract → build |
| `build_dataset.py` | Normal Python 3 | Aggregate JSON → `dataset/train.npz` + CSV |
| `config_dataset.yaml` | — | Bounds, N, seed, workers, paths |

## Parameters (Table 6 priors, uniform)

| Param | Meaning | Sample range | Passed to Abaqus |
|---|---|---|---|
| `a` | half wingspan / tip x | `[290, 310]` mm | mm |
| `b` | wingtip chord | `[20, 30]` mm | mm |
| `E1` | fuselage–wing joint Young’s modulus | `[0.5, 0.9]` × 10¹¹ Pa | **MPa** (`×1e5`) |
| `E2` | fuselage–tail joint Young’s modulus | `[0.5, 0.9]` × 10¹¹ Pa | **MPa** (`×1e5`) |

Driver stores both `E*_1e11Pa` and `E*_MPa` in `samples/lhs_*.csv`.

## One-time setup

1. Put a local (non-OneDrive) copy of the master `.cae` somewhere, e.g.
   `D:\abaqus_work\case_test-2021.cae`, and set `master_cae` in
   `config_dataset.yaml`.
2. Edit `SENSOR_COORDS` in `abaqus_run_single.py` / `extract_odb.py` if needed.
3. Install host deps: `numpy`, `scipy`, `pyyaml`, `pandas` (optional for legacy).

Smoke CAE write-inp (no solve):

```bat
cd test_run
copy ..\abaqus_run_single.py .
copy ..\..\abaqus_work\case_test-2021.cae master_model.cae
abaqus cae noGUI=abaqus_run_single.py -- 0 296.75 26.0 60000 60000 write_inp
```

Then solve that INP alone:

```bat
abaqus job=run_0000 input=run_0000.inp cpus=4 interactive
abaqus python extract_odb.py -- 0
```

## Recommended pipeline (INP + parallel solver)

```bat
python driver.py sample --n 200 --seed 42
python driver.py generate-inp
python driver.py solve-parallel --max-workers 4
python driver.py extract --max-workers 4
python driver.py build
```

Or all stages:

```bat
python driver.py all --n 200 --seed 42 --max-workers 4
```

Stages are resume-friendly: existing `.inp` / `.odb` / `status==ok` JSON are skipped.

### Layout

```text
samples/
  lhs_200_seed42.csv
  runs/run_0000/
    master_model.cae
    abaqus_run_single.py
    run_0000.inp
    run_0000.odb
    result_0000.json
dataset/
  train.npz          # theta, freqs_hz, mode_shapes, run_id, meta
  train.csv
  manifest.json
```

### `train.npz` schema

| Array | Shape | Content |
|---|---|---|
| `theta` | `(N_ok, 4)` | `[a_mm, b_mm, E1_1e11Pa, E2_1e11Pa]` |
| `freqs_hz` | `(N_ok, 7)` | first 7 flexible modes |
| `mode_shapes` | `(N_ok, 7, n_sensors)` | sensor DOF samples |
| `run_id` | `(N_ok,)` | index |

## Parallelism notes

- **generate-inp** uses CAE; keep `gen_inp_workers: 1` (or 2 if licenses allow).
- **solve-parallel** is the heavy win: `abaqus job=...` with no CAE.
  Cap `max_workers` by available Abaqus tokens and RAM (`cpus_per_job` each).
- Old all-in-one serial path: `python driver.py legacy-full`.

## Guarantees / caveats

- Master `.cae` is copied per run and never saved from CAE.
- Rigid-body / near-zero modes (`f < 0.1` Hz) are discarded; 7 flexible modes kept.
- Sensor values use nearest-node lookup; see `sensor_node_distances_mm` in JSON.
- Failed samples keep `status != ok` and do not block later stages.

## Surrogate GP (`surrogate/`)

After you have `dataset/train.npz` (e.g. N=200 FE campaign):

```bat
python surrogate\train_gp.py
python surrogate\predict_gp.py --a 300 --b 25 --E1 0.6 --E2 0.7 --return-std
```

Or: `surrogate\train_after_data.bat`. See [surrogate/README.md](surrogate/README.md).

R² / parity plots (after training or standalone):

```bat
python surrogate\plot_r2.py
```

Outputs: `surrogate/parity_r2.png`, `surrogate/r2_by_mode.png`, `surrogate/r2_scores.json`.

