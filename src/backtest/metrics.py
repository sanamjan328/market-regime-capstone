"""Performance metrics including Deflated Sharpe helpers (see eval.dsr)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe(returns: pd.Series, periods: int = 252) -> float:
    r = returns.dropna()
    if r.std() == 0 or len(r) < 2:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(periods))


def sortino(returns: pd.Series, periods: int = 252) -> float:
    r = returns.dropna()
    downside = r[r < 0]
    if len(downside) < 2 or downside.std() == 0:
        return 0.0
    return float(r.mean() / downside.std() * np.sqrt(periods))


def max_drawdown(returns: pd.Series) -> tuple[float, int]:
    cum = (1 + returns.fillna(0)).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1
    mdd = float(dd.min())
    # duration: longest consecutive drawdown streak
    in_dd = dd < 0
    longest = cur = 0
    for flag in in_dd:
        if flag:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return mdd, int(longest)


def calmar(returns: pd.Series, periods: int = 252) -> float:
    ann = float(returns.mean() * periods)
    mdd, _ = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    return ann / abs(mdd)


def hit_rate(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return 0.0
    return float((r > 0).mean())


def profit_factor(returns: pd.Series) -> float:
    r = returns.dropna()
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return np.inf if gains > 0 else 0.0
    return float(gains / losses)


def summarise(returns: pd.Series, gross: pd.Series | None = None) -> dict:
    mdd, dd_dur = max_drawdown(returns)
    out = {
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "calmar": calmar(returns),
        "max_dd": mdd,
        "dd_duration": dd_dur,
        "hit_rate": hit_rate(returns),
        "profit_factor": profit_factor(returns),
        "ann_return": float(returns.mean() * 252),
        "ann_vol": float(returns.std() * np.sqrt(252)),
        "turnover": float(returns.index.to_series().diff().count()),  # placeholder overwritten by engine
    }
    if gross is not None:
        out["gross_sharpe"] = sharpe(gross)
        out["cost_drag"] = float(gross.sum() - returns.sum())
    return out
