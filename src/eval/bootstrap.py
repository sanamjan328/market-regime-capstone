"""Stationary bootstrap (Politis–Romano) confidence intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.metrics import sharpe


def stationary_bootstrap_indices(n: int, mean_block: int, rng: np.random.Generator) -> np.ndarray:
    """Politis–Romano stationary bootstrap index path of length n."""
    p = 1.0 / max(mean_block, 1)
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(0, n)
    for t in range(1, n):
        if rng.random() < p:
            idx[t] = rng.integers(0, n)
        else:
            idx[t] = (idx[t - 1] + 1) % n
    return idx


def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_boot: int = 1000,
    mean_block: int = 10,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    r = returns.dropna().values.astype(float)
    n = len(r)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = stationary_bootstrap_indices(n, mean_block, rng)
        sample = pd.Series(r[idx])
        stats.append(sharpe(sample))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {
        "sharpe": sharpe(pd.Series(r)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": n_boot,
    }


def monte_carlo_drawdown(
    trade_returns: pd.Series,
    n_sims: int = 1000,
    seed: int = 42,
) -> dict:
    """Shuffle trade order to get a max-DD null distribution."""
    r = trade_returns.dropna().values.astype(float)
    rng = np.random.default_rng(seed)
    realised = _max_dd(r)
    null = []
    for _ in range(n_sims):
        null.append(_max_dd(rng.permutation(r)))
    null = np.asarray(null)
    return {
        "realised_max_dd": float(realised),
        "null_mean": float(null.mean()),
        "percentile": float((null <= realised).mean()),
    }


def _max_dd(r: np.ndarray) -> float:
    cum = np.cumprod(1 + r)
    peak = np.maximum.accumulate(cum)
    return float((cum / peak - 1).min())
