"""Feature engineering for regime detection and forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fracdiff import choose_d_adf, frac_diff
from .targets import triple_barrier_labels, vol_normalised_forward_return


def _shift_rolling(s: pd.Series, window: int, func: str = "mean") -> pd.Series:
    """Rolling stat using only past data at t (via shift(1))."""
    r = getattr(s.rolling(window, min_periods=max(2, window // 2)), func)()
    return r.shift(1)


def parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    rs = (np.log(high / low) ** 2) / (4.0 * np.log(2.0))
    return np.sqrt(_shift_rolling(rs, window, "mean") * 252.0)


def garman_klass_vol(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / open_) ** 2
    rs = 0.5 * log_hl - (2.0 * np.log(2.0) - 1.0) * log_co
    return np.sqrt(_shift_rolling(rs, window, "mean") * 252.0)


def yang_zhang_vol(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    log_oc = np.log(open_ / close.shift(1))
    log_cc = np.log(close / close.shift(1))
    log_ho = np.log(high / open_)
    log_lo = np.log(low / open_)
    log_co = np.log(close / open_)
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    overnight = _shift_rolling(log_oc**2, window, "mean")
    close_var = _shift_rolling(log_cc**2, window, "mean")
    rs_var = _shift_rolling(rs, window, "mean")
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    return np.sqrt((overnight + k * close_var + (1 - k) * rs_var) * 252.0)


def amihud_illiquidity(returns: pd.Series, volume: pd.Series, dollar_vol: pd.Series, window: int = 20) -> pd.Series:
    illiq = (returns.abs() / dollar_vol.replace(0, np.nan))
    return _shift_rolling(illiq, window, "mean")


def roll_spread(close: pd.Series, window: int = 20) -> pd.Series:
    """Roll (1984) spread estimator from serial covariance of price changes."""
    dp = close.diff()
    cov = dp.rolling(window).cov(dp.shift(1)).shift(1)
    spread = 2.0 * np.sqrt(np.maximum(-cov, 0.0))
    return spread


def build_feature_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Build forecasting + regime feature matrix from an OHLCV (+ macro/cross-asset) panel.

    Golden rule: every feature at t uses only data available at or before t.
    Rolling statistics use `.rolling()` then `.shift(1)`.
    """
    df = panel.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["volume"]

    out = pd.DataFrame(index=df.index)
    log_ret = np.log(close / close.shift(1))
    out["r_1"] = log_ret.shift(0)  # known at close of t; for model window we still lag targets
    # overnight / intraday split (aligned causally via shift on overnight)
    out["r_intraday"] = np.log(close / open_)
    out["r_overnight"] = np.log(open_ / close.shift(1))

    out["realized_vol_20d"] = np.sqrt(252) * _shift_rolling(log_ret, 20, "std")
    out["vol_of_vol_60d"] = _shift_rolling(out["realized_vol_20d"], 60, "std")
    out["skew_60d"] = _shift_rolling(log_ret, 60, "skew")
    out["kurt_60d"] = log_ret.rolling(60, min_periods=30).kurt().shift(1)
    out["parkinson_20d"] = parkinson_vol(high, low, 20)
    out["garman_klass_20d"] = garman_klass_vol(open_, high, low, close, 20)
    out["yang_zhang_20d"] = yang_zhang_vol(open_, high, low, close, 20)

    sma20 = close.rolling(20).mean().shift(1)
    sma100 = close.rolling(100).mean().shift(1)
    out["trend_strength"] = (sma20 - sma100) / out["realized_vol_20d"].replace(0, np.nan)

    # Momentum (vol-normalised)
    mom_12_1 = close.shift(21) / close.shift(252) - 1.0
    out["mom_12_1"] = (mom_12_1 / out["realized_vol_20d"]).replace([np.inf, -np.inf], np.nan)
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    rs = _shift_rolling(up, 14, "mean") / _shift_rolling(down, 14, "mean").replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = (ema12 - ema26).shift(1)
    out["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / out["realized_vol_20d"]

    dollar = (close * volume).replace(0, np.nan)
    out["amihud_20d"] = amihud_illiquidity(log_ret, volume, dollar, 20)
    out["volume_z_20d"] = (
        (volume - _shift_rolling(volume, 20, "mean")) / _shift_rolling(volume, 20, "std")
    )
    out["roll_spread_20d"] = roll_spread(close, 20)

    # Cross-asset
    if "tlt_close" in df.columns:
        tlt_r = np.log(df["tlt_close"] / df["tlt_close"].shift(1))
        out["corr_tlt_spy_60d"] = log_ret.rolling(60).corr(tlt_r).shift(1)
    if "uup_close" in df.columns:
        out["dxy_ret"] = np.log(df["uup_close"] / df["uup_close"].shift(1))
    if "hyg_close" in df.columns and "lqd_close" in df.columns:
        out["hyg_lqd_spread"] = np.log(df["hyg_close"] / df["lqd_close"]).diff().shift(0)
        out["hyg_lqd_spread"] = out["hyg_lqd_spread"]  # contemporaneous close; still same-day

    # Calendar (known at t)
    idx = pd.DatetimeIndex(df.index)
    out["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 5)
    out["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 5)
    out["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * idx.month / 12)

    # Macro z-scores (already PIT-lagged in data layer)
    if "T10Y2Y" in df.columns:
        s = df["T10Y2Y"]
        out["term_spread_z"] = (s - _shift_rolling(s, 252, "mean")) / _shift_rolling(s, 252, "std")
    if "BAMLC0A0CM" in df.columns:
        s = df["BAMLC0A0CM"]
        out["credit_spread_z"] = (s - _shift_rolling(s, 252, "mean")) / _shift_rolling(s, 252, "std")
    if "VIXCLS" in df.columns:
        out["vix"] = df["VIXCLS"]
        out["vix_z"] = (df["VIXCLS"] - _shift_rolling(df["VIXCLS"], 252, "mean")) / _shift_rolling(
            df["VIXCLS"], 252, "std"
        )
    if "TEDRATE" in df.columns:
        out["ted"] = df["TEDRATE"]

    # Fractional differencing on close (memory-preserving stationarity)
    d = choose_d_adf(np.log(close))
    out["close_fracdiff"] = frac_diff(np.log(close).values, d=d)
    out.attrs["fracdiff_d"] = d

    # Targets
    out["y_vol_norm"] = vol_normalised_forward_return(close, out["realized_vol_20d"], horizon=1)
    out["y_triple_barrier"] = triple_barrier_labels(close, out["realized_vol_20d"] / np.sqrt(252))
    out["fwd_ret_1"] = log_ret.shift(-1)
    out["open_next"] = open_.shift(-1)  # for next-open execution research

    return out


REGIME_FEATURE_DEFAULTS = [
    "realized_vol_20d",
    "vol_of_vol_60d",
    "skew_60d",
    "kurt_60d",
    "trend_strength",
    "term_spread_z",
]
