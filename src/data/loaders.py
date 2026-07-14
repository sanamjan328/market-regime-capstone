"""Market & macro data downloaders with local raw caching."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf


def _cache_path(raw_dir: Path, name: str) -> Path:
    return raw_dir / f"{name}.parquet"


def download_ohlcv(
    ticker: str,
    start: str,
    end: str,
    raw_dir: str | Path,
    interval: str = "1d",
    force: bool = False,
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(raw_dir, f"{ticker.replace('-', '_').lower()}_{interval}_ohlcv")
    if cache.exists() and not force:
        df = pd.read_parquet(cache)
        df.index = pd.to_datetime(df.index)
        return df

    data = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0].lower() for c in data.columns]
    else:
        data.columns = [str(c).lower() for c in data.columns]

    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in data.columns]
    data = data[keep].dropna(how="all")
    data.index = pd.to_datetime(data.index).tz_localize(None)
    data.index.name = "date"
    data.to_parquet(cache)
    return data


def download_cross_asset(
    tickers: Iterable[str],
    start: str,
    end: str,
    raw_dir: str | Path,
    force: bool = False,
) -> pd.DataFrame:
    frames = []
    for t in tickers:
        df = download_ohlcv(t, start, end, raw_dir, force=force)
        if "close" not in df.columns:
            continue
        frames.append(df["close"].rename(f"{t.lower()}_close"))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1).sort_index()
    cache = _cache_path(Path(raw_dir), "cross_asset_close")
    out.to_parquet(cache)
    return out


def download_fred(
    series_ids: Iterable[str],
    start: str,
    end: str,
    raw_dir: str | Path,
    force: bool = False,
) -> pd.DataFrame:
    """
    Pull FRED series via pandas-datareader.

    Note: full ALFRED vintage alignment is the gold standard; here we apply
    explicit lag in `pit.lag_macro_series` as the minimum PIT discipline.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(raw_dir, "fred_macro")
    if cache.exists() and not force:
        df = pd.read_parquet(cache)
        df.index = pd.to_datetime(df.index)
        return df

    try:
        import pandas_datareader.data as web
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pandas-datareader is required for FRED downloads") from exc

    frames = []
    for sid in series_ids:
        try:
            s = web.DataReader(sid, "fred", start=start, end=end)
            s.columns = [sid]
            frames.append(s)
        except Exception as exc:  # network / series availability
            print(f"[warn] FRED {sid} failed: {exc}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1).sort_index()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out.index.name = "date"
    out.to_parquet(cache)
    return out
