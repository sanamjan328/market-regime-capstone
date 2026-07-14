"""Ablation table runner — heart of the thesis evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.backtest.metrics import calmar, max_drawdown, profit_factor, sharpe, sortino


@dataclass
class AblationRow:
    model: str
    purpose: str
    sharpe: float
    sortino: float
    calmar: float
    max_dd: float
    dd_duration: int
    hit_rate: float
    profit_factor: float
    ann_return: float
    ann_vol: float
    n_obs: int
    gross_sharpe: float = field(default=float("nan"))
    cost_drag: float = field(default=float("nan"))


ABLATION_SPEC = [
    ("buy_hold", "Did you beat doing nothing?"),
    ("zero", "Is there any signal?"),
    ("arima_garch", "Classical econometrics benchmark"),
    ("dlinear", "Linear model that embarrasses Transformers"),
    ("lightgbm", "Trees usually win on tabular finance"),
    ("transformer_no_regime", "KEY CONTROL"),
    ("transformer_film", "THESIS"),
    ("transformer_hard_switch", "Soft vs hard conditioning"),
]


def directional_hit(mu: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(mu) & np.isfinite(y)
    if mask.sum() == 0:
        return 0.0
    return float((np.sign(mu[mask]) == np.sign(y[mask])).mean())


def strategy_returns_from_mu(mu: np.ndarray, fwd_ret: np.ndarray) -> pd.Series:
    pos = np.clip(mu, -1, 1)
    return pd.Series(pos * fwd_ret)


def build_ablation_table(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    predictions: model_name -> DataFrame with columns mu, sigma, y, and optionally fwd_ret.

    Uses raw mu-clipped positions (no Kelly sizing) so all models are compared
    on an identical signal-only basis before cost/sizing enters in backtest.py.
    """
    rows = []
    purpose = {m: p for m, p in ABLATION_SPEC}
    for name, df in predictions.items():
        y = df["y"].values
        mu = df["mu"].values
        fwd = df["fwd_ret"].values if "fwd_ret" in df.columns else y
        rets = strategy_returns_from_mu(mu, fwd)
        mdd, dd_dur = max_drawdown(rets)
        rows.append(
            AblationRow(
                model=name,
                purpose=purpose.get(name, ""),
                sharpe=sharpe(rets),
                sortino=sortino(rets),
                calmar=calmar(rets),
                max_dd=mdd,
                dd_duration=int(dd_dur),
                hit_rate=directional_hit(mu, y),
                profit_factor=profit_factor(rets),
                ann_return=float(rets.mean() * 252),
                ann_vol=float(rets.std() * np.sqrt(252)),
                n_obs=int(np.isfinite(y).sum()),
            )
        )
    return pd.DataFrame([r.__dict__ for r in rows])
