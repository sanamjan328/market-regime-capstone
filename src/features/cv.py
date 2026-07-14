"""Purged expanding walk-forward CV with embargo (Lopez de Prado)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def purged_walk_forward(
    dates: pd.DatetimeIndex,
    initial_train_end: str = "2010-12-31",
    test_years: int = 2,
    embargo_days: int = 20,
    purge_horizon_days: int = 1,
    final_holdout_start: str | None = "2022-01-01",
) -> Iterator[Fold]:
    """
    Expanding-window walk-forward with purge + embargo.

    - Purge: drop train samples whose label horizon overlaps the test set.
    - Embargo: buffer of `embargo_days` after train (before test) to kill
      serial-correlation leakage.
    - Folds stop before `final_holdout_start` so the holdout is untouched.
    """
    dates = pd.DatetimeIndex(dates).sort_values()
    n = len(dates)
    train_end = pd.Timestamp(initial_train_end)
    holdout_start = pd.Timestamp(final_holdout_start) if final_holdout_start else None
    fold_id = 0

    while True:
        test_start = train_end + timedelta(days=1 + embargo_days)
        test_end = train_end + pd.DateOffset(years=test_years)
        if holdout_start is not None and test_start >= holdout_start:
            break
        if holdout_start is not None and test_end > holdout_start:
            test_end = holdout_start - timedelta(days=1)

        train_mask = dates <= train_end
        # purge: remove samples within purge_horizon of the (embargoed) test boundary
        purge_cut = train_end - timedelta(days=purge_horizon_days)
        train_mask &= dates <= purge_cut
        test_mask = (dates >= test_start) & (dates <= test_end)
        if holdout_start is not None:
            test_mask &= dates < holdout_start

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        if len(train_idx) == 0 or len(test_idx) == 0:
            break

        yield Fold(
            fold_id=fold_id,
            train_idx=train_idx,
            test_idx=test_idx,
            train_end=train_end,
            test_start=pd.Timestamp(dates[test_idx[0]]),
            test_end=pd.Timestamp(dates[test_idx[-1]]),
        )
        fold_id += 1
        train_end = test_end
        if train_end >= dates[-1]:
            break
        # safety
        if fold_id > 20:
            break

    _ = n  # silence unused in some analyses


def holdout_indices(
    dates: pd.DatetimeIndex,
    start: str = "2022-01-01",
    end: str = "2024-12-31",
) -> np.ndarray:
    dates = pd.DatetimeIndex(dates)
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return np.where(mask)[0]
