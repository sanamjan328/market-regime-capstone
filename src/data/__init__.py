"""Data loaders, point-in-time alignment, and parquet I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .loaders import download_cross_asset, download_fred, download_ohlcv
from .pit import lag_macro_series


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_primary_panel(
    universe: str = "SPY",
    start: str = "2000-01-01",
    end: str = "2024-12-31",
    cross_asset: Iterable[str] | None = None,
    fred_series: dict | None = None,
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    macro_lag_days_default: int = 1,
) -> pd.DataFrame:
    """
    Download OHLCV + cross-asset + FRED macros, apply PIT lags, write parquet.

    Macro series are lagged by their configured publication delay so that
    features at time t only use information available on or before t.
    """
    raw_dir = _ensure_dir(Path(raw_dir))
    processed_dir = _ensure_dir(Path(processed_dir))
    cross_asset = list(cross_asset or ["TLT", "GLD", "UUP", "HYG", "LQD"])
    fred_series = fred_series or {
        "VIXCLS": {"lag_days": 1},
        "T10Y2Y": {"lag_days": 1},
        "BAMLC0A0CM": {"lag_days": 1},
        "TEDRATE": {"lag_days": 1},
    }

    ohlcv = download_ohlcv(universe, start, end, raw_dir)
    xa = download_cross_asset(cross_asset, start, end, raw_dir)
    macro = download_fred(list(fred_series.keys()), start, end, raw_dir)

    # Point-in-time: lag each macro by its release delay
    lagged_macros = []
    for col in macro.columns:
        lag = int(fred_series.get(col, {}).get("lag_days", macro_lag_days_default))
        lagged_macros.append(lag_macro_series(macro[col], lag_days=lag).rename(col))
    macro_pit = pd.concat(lagged_macros, axis=1) if lagged_macros else pd.DataFrame()

    panel = ohlcv.join(xa, how="left").join(macro_pit, how="left")
    panel = panel.sort_index().ffill()  # forward-fill known (already lagged) macros
    out = processed_dir / f"{universe.lower()}_panel.parquet"
    panel.to_parquet(out)
    return panel


def build_btc_panel(
    start: str = "2018-01-01",
    end: str = "2024-12-31",
    interval: str = "1h",
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
) -> pd.DataFrame:
    """Secondary universe for robustness (BTC-USD)."""
    raw_dir = _ensure_dir(Path(raw_dir))
    processed_dir = _ensure_dir(Path(processed_dir))
    ohlcv = download_ohlcv("BTC-USD", start, end, raw_dir, interval=interval)
    out = processed_dir / "btc_panel.parquet"
    ohlcv.to_parquet(out)
    return ohlcv


def load_panel(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)
