"""Alternative regime models for the comparison section."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fit_markov_switching_variance(
    returns: pd.Series,
    k_regimes: int = 3,
) -> pd.DataFrame:
    """
    Hamilton-style Markov-switching regression on returns (variance switching).
    Returns filtered / smoothed regime probabilities from statsmodels.
    Prefer filtered; document if smoothed is used for comparison only.
    """
    try:
        import statsmodels.api as sm
    except ImportError as exc:  # pragma: no cover
        raise ImportError("statsmodels required") from exc

    y = returns.dropna()
    mod = sm.tsa.MarkovRegression(
        y, k_regimes=k_regimes, trend="c", switching_variance=True
    )
    res = mod.fit(search_reps=10, disp=False)
    # filtered probs if available
    if hasattr(res, "filtered_marginal_probabilities"):
        probs = res.filtered_marginal_probabilities
    else:
        probs = res.smoothed_marginal_probabilities
    return pd.DataFrame(probs, index=y.index)


def fit_pelt_changepoints(X: np.ndarray, pen: float = 10.0) -> list[int]:
    """Non-parametric PELT change-points (ruptures) as sanity check."""
    try:
        import ruptures as rpt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("ruptures required") from exc

    algo = rpt.Pelt(model="rbf").fit(X)
    return algo.predict(pen=pen)


def wasserstein_kmeans(
    returns: pd.Series,
    window: int = 60,
    n_clusters: int = 3,
    n_quantiles: int = 21,
    max_iter: int = 50,
    random_state: int = 0,
) -> np.ndarray:
    """
    Cluster rolling return distributions using Wasserstein-1 on quantile grids.
    Returns hard labels aligned to the end of each window (NaNs for warm-up).
    """
    rng = np.random.default_rng(random_state)
    r = returns.values.astype(float)
    n = len(r)
    qs = np.linspace(0.05, 0.95, n_quantiles)
    features = []
    idxs = []
    for t in range(window, n):
        w = r[t - window : t]
        if np.any(~np.isfinite(w)):
            continue
        features.append(np.quantile(w, qs))
        idxs.append(t)
    if not features:
        return np.full(n, np.nan)
    F = np.asarray(features)
    # init centers
    centers = F[rng.choice(len(F), size=n_clusters, replace=False)]
    labels = np.zeros(len(F), dtype=int)
    for _ in range(max_iter):
        # assign by L1 on quantile grid ≈ W1 for 1D
        d = np.abs(F[:, None, :] - centers[None, :, :]).mean(axis=2)
        labels = d.argmin(axis=1)
        new_centers = np.vstack(
            [
                F[labels == k].mean(axis=0) if np.any(labels == k) else centers[k]
                for k in range(n_clusters)
            ]
        )
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    out = np.full(n, np.nan)
    for i, t in enumerate(idxs):
        out[t] = labels[i]
    return out


def try_hsmm_note() -> str:
    """
    pyhsmm is intentionally timeboxed. If unavailable, fall back to Markov-Switching AR.
    """
    try:
        import pyhsmm  # noqa: F401
        return "pyhsmm available"
    except Exception:
        return (
            "pyhsmm unavailable or awkward deps — use statsmodels.MarkovRegression "
            "as duration-aware comparison (documented substitution)."
        )
