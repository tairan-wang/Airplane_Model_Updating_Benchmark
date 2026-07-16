@echo off
REM Run after FE pipeline finishes (dataset\train.npz present)
cd /d %~dp0\..
if not exist dataset\train.npz (
  echo dataset\train.npz not found yet. Wait for make_training_data to finish.
  exit /b 1
)
python surrogate\train_gp.py
echo.
echo Done. See surrogate\metrics.json
