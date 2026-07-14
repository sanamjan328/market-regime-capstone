"""Transaction cost models."""

from __future__ import annotations

import numpy as np


def commission_cost(turnover: np.ndarray, commission_bps: float = 2.0) -> np.ndarray:
    return np.abs(turnover) * (commission_bps / 1e4)


def impact_cost(
    turnover: np.ndarray,
    volume: np.ndarray,
    adv: np.ndarray,
    k: float = 0.1,
) -> np.ndarray:
    """Approximate market impact ≈ k * (volume / ADV)^0.5 * |Δpos| (simplified)."""
    participation = np.abs(turnover) / np.maximum(adv, 1.0)
    return k * np.sqrt(np.maximum(participation, 0.0)) * np.abs(turnover) * 0.01


def total_cost(
    positions: np.ndarray,
    volume: np.ndarray | None = None,
    commission_bps: float = 2.0,
    slippage_k: float = 0.1,
    borrow_bps: float = 0.0,
) -> np.ndarray:
    prev = np.concatenate([[0.0], positions[:-1]])
    turnover = positions - prev
    costs = commission_cost(turnover, commission_bps)
    if volume is not None:
        adv = np.maximum(pd_rolling_mean(volume, 20), 1.0)
        costs = costs + impact_cost(turnover, volume, adv, k=slippage_k)
    # daily borrow on short notional
    shorts = np.maximum(-positions, 0.0)
    costs = costs + shorts * (borrow_bps / 1e4) / 252.0
    return costs


def pd_rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    csum = np.cumsum(np.insert(x.astype(float), 0, 0.0))
    for i in range(window - 1, len(x)):
        out[i] = (csum[i + 1] - csum[i + 1 - window]) / window
    # warm-up fill
    out = np.where(np.isfinite(out), out, np.nanmean(x))
    return out
