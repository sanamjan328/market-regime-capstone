"""Expanding-window scalers and winsorisation (fit on train only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def winsorize(
    df: pd.DataFrame,
    lower: float = 0.01,
    upper: float = 0.99,
    ref: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Clip columns to quantiles computed on `ref` (train) or on `df`."""
    base = ref if ref is not None else df
    out = df.copy()
    for c in out.columns:
        lo = base[c].quantile(lower)
        hi = base[c].quantile(upper)
        out[c] = out[c].clip(lo, hi)
    return out


class ExpandingStandardScaler:
    """
    Fit StandardScaler on train fold only; transform train/test with frozen params.
    Refit each walk-forward fold — never fit on the full history.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.columns_: list[str] | None = None

    def fit(self, X: pd.DataFrame) -> "ExpandingStandardScaler":
        self.columns_ = list(X.columns)
        self.scaler.fit(X.fillna(0.0).values)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.columns_ is None:
            raise RuntimeError("Scaler not fit")
        vals = self.scaler.transform(X[self.columns_].fillna(0.0).values)
        return pd.DataFrame(vals, index=X.index, columns=self.columns_)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
