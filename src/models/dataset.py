"""Windowed datasets for sequence models."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        regimes: np.ndarray | None,
        lookback: int,
        indices: np.ndarray | None = None,
    ):
        self.features = features.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.regimes = None if regimes is None else regimes.astype(np.float32)
        self.lookback = lookback
        if indices is None:
            indices = np.arange(len(features))
        # valid endpoints where full lookback exists and target is finite
        valid = []
        for i in indices:
            if i < lookback:
                continue
            if not np.isfinite(self.targets[i]):
                continue
            window = self.features[i - lookback : i]
            if not np.isfinite(window).all():
                continue
            valid.append(int(i))
        self.indices = np.asarray(valid, dtype=int)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        i = self.indices[idx]
        x = self.features[i - self.lookback : i]
        y = self.targets[i]
        if self.regimes is None:
            s = np.zeros(1, dtype=np.float32)
        else:
            s = self.regimes[i]
        return (
            torch.from_numpy(x),
            torch.tensor(y, dtype=torch.float32),
            torch.from_numpy(s),
        )
