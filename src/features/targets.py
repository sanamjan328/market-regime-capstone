"""Target construction: vol-normalised forward return + triple-barrier labels."""

from __future__ import annotations

import numpy as np
import pandas as pd


def vol_normalised_forward_return(
    close: pd.Series,
    realized_vol: pd.Series,
    horizon: int = 1,
) -> pd.Series:
    """
    y_t = r_{t+h} / sigma_hat_t

    Predicting raw returns is dominated by volatility clustering.
    """
    log_ret = np.log(close / close.shift(1))
    fwd = log_ret.shift(-horizon)
    y = fwd / realized_vol.replace(0.0, np.nan)
    y.name = "y_vol_norm"
    return y


def triple_barrier_labels(
    close: pd.Series,
    volatility: pd.Series,
    pt_mult: float = 1.0,
    sl_mult: float = 1.0,
    max_holding: int = 10,
) -> pd.Series:
    """
    Lopez de Prado triple-barrier labelling.
    +1 profit-take, -1 stop-loss, 0 vertical (time) barrier.
    """
    prices = close.values.astype(float)
    vols = volatility.values.astype(float)
    n = len(prices)
    labels = np.full(n, np.nan)

    for i in range(n - 1):
        if not np.isfinite(vols[i]) or vols[i] <= 0 or not np.isfinite(prices[i]):
            continue
        pt = prices[i] * (1.0 + pt_mult * vols[i])
        sl = prices[i] * (1.0 - sl_mult * vols[i])
        end = min(i + max_holding, n - 1)
        path = prices[i + 1 : end + 1]
        if path.size == 0:
            continue
        hit_pt = np.where(path >= pt)[0]
        hit_sl = np.where(path <= sl)[0]
        first_pt = hit_pt[0] if hit_pt.size else None
        first_sl = hit_sl[0] if hit_sl.size else None
        if first_pt is None and first_sl is None:
            labels[i] = 0.0
        elif first_sl is None or (first_pt is not None and first_pt <= first_sl):
            labels[i] = 1.0
        else:
            labels[i] = -1.0

    return pd.Series(labels, index=close.index, name="y_triple_barrier")
