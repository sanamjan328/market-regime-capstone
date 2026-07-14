"""Backtest package."""

from .engine import BacktestResult, cross_check_engines, event_loop_backtest
from .sizing import kelly_lite_positions, learn_regime_gates

__all__ = [
    "event_loop_backtest",
    "cross_check_engines",
    "BacktestResult",
    "kelly_lite_positions",
    "learn_regime_gates",
]
