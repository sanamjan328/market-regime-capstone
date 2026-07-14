"""Fractional differencing (Lopez de Prado, Advances in Financial ML, Ch. 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def frac_diff_weights(
    d: float,
    thresh: float = 1e-5,
    max_size: int = 100,
) -> np.ndarray:
    """Binomial expansion weights for fractional differencing."""
    w = [1.0]
    k = 1
    while k < max_size:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])  # oldest weight first


def frac_diff(
    series: np.ndarray | pd.Series,
    d: float,
    thresh: float = 1e-5,
    max_size: int = 100,
) -> np.ndarray:
    """Fixed-width-window fractional differencing (Lopez de Prado, Ch. 5)."""
    x = np.asarray(series, dtype=float)
    w = frac_diff_weights(d, thresh, max_size=max_size)
    width = len(w) - 1
    out = np.full(len(x), np.nan)
    for i in range(width, len(x)):
        out[i] = np.dot(w, x[i - width : i + 1])
    return out


def choose_d_adf(
    series: pd.Series,
    d_grid: np.ndarray | None = None,
    pvalue: float = 0.05,
) -> float:
    """
    Smallest d that passes an ADF stationarity test (maximises retained memory).
    Falls back to 0.4 if statsmodels is unavailable or no d passes.
    """
    d_grid = d_grid if d_grid is not None else np.round(np.arange(0.0, 1.05, 0.05), 2)
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return 0.4

    x = series.dropna().astype(float)
    for d in d_grid:
        y = frac_diff(x.values, float(d))
        y = y[~np.isnan(y)]
        if len(y) < 50:
            continue
        try:
            p = adfuller(y, maxlag=1, regression="c", autolag=None)[1]
        except Exception:
            continue
        if p < pvalue:
            return float(d)
    return 0.4
