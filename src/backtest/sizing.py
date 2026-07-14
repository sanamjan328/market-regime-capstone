"""Uncertainty-aware, regime-gated position sizing."""

from __future__ import annotations

import numpy as np


def kelly_lite_positions(
    mu: np.ndarray,
    sigma: np.ndarray,
    realized_vol_20d: np.ndarray,
    regime_posteriors: np.ndarray | None,
    regime_gates: dict[int, float] | None = None,
    vol_target: float = 0.10,
    leverage_cap: float = 1.0,
) -> np.ndarray:
    """
    edge = mu / sigma^2  (Kelly-lite)
    position = clip(edge) * vol_scaling * regime_gate
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    rv = np.asarray(realized_vol_20d, dtype=float)
    edge = mu / np.maximum(sigma**2, 1e-8)
    raw = np.clip(edge, -1.0, 1.0)
    scaling = vol_target / np.maximum(rv * np.sqrt(252.0), 1e-6)
    gate = np.ones(len(mu))
    if regime_posteriors is not None and regime_gates is not None:
        hard = np.argmax(regime_posteriors, axis=1)
        gate = np.array([regime_gates.get(int(k), 1.0) for k in hard], dtype=float)
    pos = raw * scaling * gate
    return np.clip(pos, -leverage_cap, leverage_cap)


def learn_regime_gates(
    regimes: np.ndarray,
    forwards: np.ndarray,
    n_states: int,
) -> dict[int, float]:
    """
    Learn simple gates on TRAIN only from per-regime mean forward returns.
    Never hand-tune on test performance.
    """
    gates = {}
    for k in range(n_states):
        mask = np.argmax(regimes, axis=1) == k
        if mask.sum() < 20:
            gates[k] = 0.5
            continue
        mu = np.nanmean(forwards[mask])
        if mu > 0.0005:
            gates[k] = 1.0
        elif mu > 0:
            gates[k] = 0.5
        elif mu > -0.0005:
            gates[k] = 0.25
        else:
            gates[k] = 0.0
    return gates
