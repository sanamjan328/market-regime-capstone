"""Per-regime P&L attribution — the proof that the regime layer does real work."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.metrics import calmar, max_drawdown, sharpe, sortino

_DEFAULT_NAMES = {
    0: "low_vol_bull",
    1: "transition",
    2: "high_vol_bear",
    3: "crisis",
}


def per_regime_attribution(
    net_returns: pd.Series,
    regime_posteriors: pd.DataFrame,
    regime_names: dict[int, str] | None = None,
) -> pd.DataFrame:
    """
    Break down strategy returns by dominant regime at each bar.

    Parameters
    ----------
    net_returns : pd.Series
        Daily net strategy returns indexed by date.
    regime_posteriors : pd.DataFrame
        Columns p_regime_0 .. p_regime_K, indexed by date.
    regime_names : optional mapping of state index -> label

    Returns
    -------
    DataFrame with one row per regime: n_bars, pct_time, ann_return,
    ann_vol, sharpe, sortino, calmar, max_dd, dd_duration, hit_rate.
    """
    names = {**_DEFAULT_NAMES, **(regime_names or {})}
    idx = net_returns.index.intersection(regime_posteriors.index)
    rets = net_returns.loc[idx]
    posts = regime_posteriors.loc[idx]
    hard = posts.values.argmax(axis=1)

    rows = []
    for k in range(posts.shape[1]):
        mask = hard == k
        if mask.sum() == 0:
            continue
        r_k = rets.iloc[mask]
        mdd, dd_dur = max_drawdown(r_k)
        rows.append(
            {
                "regime": k,
                "name": names.get(k, f"regime_{k}"),
                "n_bars": int(mask.sum()),
                "pct_time": float(mask.mean()),
                "mean_daily_ret": float(r_k.mean()),
                "ann_return": float(r_k.mean() * 252),
                "ann_vol": float(r_k.std() * np.sqrt(252)),
                "sharpe": sharpe(r_k),
                "sortino": sortino(r_k),
                "calmar": calmar(r_k),
                "max_dd": mdd,
                "dd_duration": int(dd_dur),
                "hit_rate": float((r_k > 0).mean()),
            }
        )
    return pd.DataFrame(rows)
