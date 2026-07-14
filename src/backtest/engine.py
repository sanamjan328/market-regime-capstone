"""Backtest engines: vectorised path + event-loop cross-check."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import total_cost
from .metrics import summarise


@dataclass
class BacktestResult:
    positions: pd.Series
    gross_returns: pd.Series
    net_returns: pd.Series
    costs: pd.Series
    metrics: dict


def event_loop_backtest(
    open_prices: pd.Series,
    close_prices: pd.Series,
    positions: pd.Series,
    volume: pd.Series | None = None,
    commission_bps: float = 2.0,
    slippage_k: float = 0.1,
    borrow_bps: float = 0.0,
) -> BacktestResult:
    """
    Execute at the NEXT bar's open: signal at t close -> trade at t+1 open.
    Returns from open-to-open (or open-to-close proxy via close returns shifted).
    """
    pos = positions.astype(float).copy()
    # delay position by 1: decide at t, hold over t+1
    executed = pos.shift(1).fillna(0.0)
    # use close-to-close as bar return proxy after next-open entry approximation
    bar_ret = close_prices.pct_change().fillna(0.0)
    gross = executed * bar_ret
    costs = pd.Series(
        total_cost(
            executed.values,
            volume.values if volume is not None else None,
            commission_bps=commission_bps,
            slippage_k=slippage_k,
            borrow_bps=borrow_bps,
        ),
        index=executed.index,
    )
    net = gross - costs
    turnover = executed.diff().abs().fillna(0.0)
    metrics = summarise(net, gross)
    metrics["turnover"] = float(turnover.mean() * 252)
    return BacktestResult(
        positions=executed,
        gross_returns=gross,
        net_returns=net,
        costs=costs,
        metrics=metrics,
    )


def vectorbt_backtest(
    close: pd.Series,
    positions: pd.Series,
    commission_bps: float = 2.0,
) -> BacktestResult | None:
    """Optional vectorbt cross-check; returns None if vectorbt missing."""
    try:
        import vectorbt as vbt
    except ImportError:
        return None

    # map continuous positions to size updates
    size = positions.fillna(0.0)
    pf = vbt.Portfolio.from_orders(
        close=close,
        size=size.diff().fillna(size),
        size_type="targetpercent",
        fees=commission_bps / 1e4,
        freq="1D",
    )
    rets = pf.returns()
    metrics = {
        "sharpe": float(pf.sharpe_ratio()) if hasattr(pf, "sharpe_ratio") else float("nan"),
        "max_dd": float(pf.max_drawdown()) if hasattr(pf, "max_drawdown") else float("nan"),
        "ann_return": float(pf.annualized_return()) if hasattr(pf, "annualized_return") else float("nan"),
    }
    return BacktestResult(
        positions=size,
        gross_returns=rets,
        net_returns=rets,
        costs=pd.Series(0.0, index=rets.index),
        metrics=metrics,
    )


def cross_check_engines(
    close: pd.Series,
    positions: pd.Series,
    commission_bps: float = 2.0,
    atol: float = 5e-3,
) -> dict:
    """Run event loop and vectorbt; flag disagreement (possible lookahead)."""
    ev = event_loop_backtest(close, close, positions, commission_bps=commission_bps)
    vb = vectorbt_backtest(close, positions, commission_bps=commission_bps)
    out = {"event_sharpe": ev.metrics["sharpe"], "vectorbt": None, "agree": True}
    if vb is None:
        out["vectorbt"] = "unavailable"
        return out
    out["vectorbt_sharpe"] = vb.metrics.get("sharpe")
    if out["vectorbt_sharpe"] is not None and np.isfinite(out["vectorbt_sharpe"]):
        out["agree"] = abs(out["event_sharpe"] - out["vectorbt_sharpe"]) < atol * 10
    return out
