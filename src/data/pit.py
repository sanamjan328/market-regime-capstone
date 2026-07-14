"""Point-in-time (PIT) alignment helpers for revised macro series."""

from __future__ import annotations

import pandas as pd


def lag_macro_series(series: pd.Series, lag_days: int = 1) -> pd.Series:
    """
    Shift a macro series forward by `lag_days` calendar/business observations.

    Documentation: FRED macros are revised; the value downloaded today for a
    historical reference date is not the value available on that date. Lagging
    by typical publication delay is the minimum PIT discipline (ALFRED vintages
    preferred for full rigor).
    """
    if lag_days < 0:
        raise ValueError("lag_days must be >= 0")
    out = series.shift(lag_days)
    out.name = series.name
    return out


def asof_align(
    left: pd.DataFrame,
    right: pd.DataFrame,
    direction: str = "backward",
) -> pd.DataFrame:
    """Align lower-frequency macros onto a trading calendar via merge_asof."""
    left = left.sort_index().reset_index()
    right = right.sort_index().reset_index()
    date_col_l = left.columns[0]
    date_col_r = right.columns[0]
    merged = pd.merge_asof(
        left,
        right,
        left_on=date_col_l,
        right_on=date_col_r,
        direction=direction,
    )
    return merged.set_index(date_col_l).drop(columns=[date_col_r], errors="ignore")
