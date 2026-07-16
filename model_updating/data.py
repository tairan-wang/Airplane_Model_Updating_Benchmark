"""
data.py -- standardisation and dataloaders shared by all five methods.
Both theta and y are z-scored; scalers are stored with every checkpoint so
sampling can map back to physical units.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class Standardiser:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardiser":
        return cls(x.mean(axis=0), x.std(axis=0) + 1e-12)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return z * self.std + self.mean

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state(cls, d: dict) -> "Standardiser":
        return cls(np.asarray(d["mean"]), np.asarray(d["std"]))


def load_dataset(path: Path, val_frac: float = 0.1, seed: int = 0):
    d = np.load(path, allow_pickle=True)
    theta, y = d["theta"].astype(np.float32), d["y"].astype(np.float32)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(theta))
    n_val = int(val_frac * len(theta))
    va, tr = idx[:n_val], idx[n_val:]

    s_theta = Standardiser.fit(theta[tr])
    s_y = Standardiser.fit(y[tr])

    def mk(ids):
        return (torch.from_numpy(s_theta.transform(theta[ids]).astype(np.float32)),
                torch.from_numpy(s_y.transform(y[ids]).astype(np.float32)))

    return mk(tr), mk(va), s_theta, s_y


def loaders(train, val, batch_size: int):
    tr = DataLoader(TensorDataset(*train), batch_size=batch_size,
                    shuffle=True, drop_last=True)
    va = DataLoader(TensorDataset(*val), batch_size=4096, shuffle=False)
    return tr, va
