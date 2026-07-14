"""Regime validation helpers: summaries, overlays, stability, exogenous checks."""

from __future__ import annotations

import numpy as np
import pandas as pd


def per_regime_summary(
    returns: pd.Series,
    hard_states: np.ndarray,
    n_states: int,
) -> pd.DataFrame:
    rows = []
    for k in range(n_states):
        mask = hard_states == k
        r = returns[mask]
        if len(r) == 0:
            rows.append(
                {
                    "state": k,
                    "n": 0,
                    "mean_ret": np.nan,
                    "ann_vol": np.nan,
                    "sharpe": np.nan,
                    "max_dd": np.nan,
                    "mean_duration": np.nan,
                }
            )
            continue
        cum = (1 + r).cumprod()
        peak = cum.cummax()
        dd = (cum / peak - 1).min()
        mu = r.mean() * 252
        vol = r.std() * np.sqrt(252)
        # duration: run lengths
        durations = []
        run = 0
        for s in hard_states:
            if s == k:
                run += 1
            elif run > 0:
                durations.append(run)
                run = 0
        if run > 0:
            durations.append(run)
        rows.append(
            {
                "state": k,
                "n": int(mask.sum()),
                "mean_ret": float(mu),
                "ann_vol": float(vol),
                "sharpe": float(mu / vol) if vol > 0 else np.nan,
                "max_dd": float(dd),
                "mean_duration": float(np.mean(durations)) if durations else np.nan,
            }
        )
    return pd.DataFrame(rows)


def viterbi_hard_states(filtered: np.ndarray) -> np.ndarray:
    """Argmax of filtered posteriors (causal hard labels)."""
    return np.argmax(filtered, axis=1)


def confusion_vs_exogenous(
    hard_states: np.ndarray,
    crisis_mask: np.ndarray,
    crisis_state: int | None = None,
) -> dict:
    """
    Cross-check high-vol / crisis state against an exogenous crisis mask
    (e.g. VIX > 30 or NBER recession dates mapped to trading days).
    """
    if crisis_state is None:
        # assume highest-index state is crisis after vol canonization
        crisis_state = int(hard_states.max())
    pred = hard_states == crisis_state
    tp = int((pred & crisis_mask).sum())
    fp = int((pred & ~crisis_mask).sum())
    fn = int((~pred & crisis_mask).sum())
    tn = int((~pred & ~crisis_mask).sum())
    # Cohen's kappa
    n = tp + fp + fn + tn
    po = (tp + tn) / n if n else 0.0
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (n**2) if n else 0.0
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "cohen_kappa": float(kappa)}


def stability_jaccard(
    states_a: np.ndarray,
    states_b: np.ndarray,
) -> float:
    """Overlap of crisis/high-vol state membership after aligning lengths."""
    n = min(len(states_a), len(states_b))
    a = states_a[-n:]
    b = states_b[-n:]
    # compare argmax equality rate
    return float((a == b).mean())
