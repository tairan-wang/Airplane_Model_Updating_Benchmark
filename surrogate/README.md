# Surrogate folder — multi-output GP for 7 natural frequencies

## What lives here

| File | Role |
|------|------|
| `make_training_data.py` | Call parent `driver.py` to make `N=200` FE dataset |
| `train_gp.py` | Fit multi-output GP (`sklearn` MultiOutputRegressor) |
| `predict_gp.py` | Predict frequencies (+ optional std) |
| `config.yaml` | Paths, split, GP knobs |
| `multioutput_gp.joblib` | Saved model (after training) |
| `input_scaler.joblib` | Input StandardScaler |
| `metrics.json` | Hold-out errors |

## Multi-output GP

One independent `GaussianProcessRegressor` per mode (`f1…f7`), wrapped in
`MultiOutputRegressor`. Inputs are `(a_mm, b_mm, E1_1e11Pa, E2_1e11Pa)` after
standard scaling. This is the usual practical “multi-output GP” for modal
surrogates (not NSGA-style multi-objective optimisation).

## Workflow

```bat
cd D:\python_run_package
pip install -r surrogate\requirements.txt

REM 1) Generate 200 FE samples (LONG — hours to days)
python surrogate\make_training_data.py --n 200 --seed 42 --max-workers 4

REM 2) Train GP once dataset\train.npz exists
python surrogate\train_gp.py

REM 3) Predict
python surrogate\predict_gp.py --a 300 --b 25 --E1 0.6 --E2 0.7 --return-std
```

Sample LHS only (no Abaqus):

```bat
python surrogate\make_training_data.py --n 200 --sample-only
```

## Timing note

`make_training_data` without `--sample-only` runs CAE write-inp + parallel
solver for 200 points. Wall-clock is typically **days**, not minutes. Leave it
running overnight; stages resume if interrupted (`driver.py` skips finished
runs).
