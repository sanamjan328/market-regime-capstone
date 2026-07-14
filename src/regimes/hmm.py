"""Gaussian HMM regime detection with per-fold refit and causal filtering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from .filtering import canonicalize_states, filtered_posteriors


@dataclass
class RegimeFitResult:
    k: int
    bic: float
    mean_duration: float
    model: GaussianHMM
    filtered: np.ndarray  # (T, K) on the window passed to filter


def mean_state_duration(transmat: np.ndarray) -> float:
    """Expected geometric duration averaged across states."""
    diag = np.clip(np.diag(transmat), 1e-6, 1 - 1e-6)
    durations = 1.0 / (1.0 - diag)
    return float(np.mean(durations))


def fit_gaussian_hmm(
    X: np.ndarray,
    n_components: int = 3,
    random_state: int = 0,
    n_iter: int = 200,
    vol_index: int = 0,
) -> GaussianHMM:
    model = GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
        verbose=False,
    )
    model.fit(X)
    canonicalize_states(model, vol_index=vol_index)
    return model


def select_k(
    X_train: np.ndarray,
    k_grid: list[int] | None = None,
    min_duration: float = 10.0,
    random_state: int = 0,
    vol_index: int = 0,
) -> tuple[int, dict[int, dict]]:
    """
    Select K by BIC + persistence (>= min_duration days) jointly.
    Among candidates that pass persistence, pick lowest BIC.
    """
    k_grid = k_grid or [2, 3, 4, 5, 6]
    scores: dict[int, dict] = {}
    for k in k_grid:
        model = fit_gaussian_hmm(X_train, n_components=k, random_state=random_state, vol_index=vol_index)
        ll = model.score(X_train)
        n, d = X_train.shape
        # full covariance params: K * (d + d(d+1)/2) + K*(K-1) + (K-1)
        n_cov = d * (d + 1) / 2
        n_params = k * (d + n_cov) + k * (k - 1) + (k - 1)
        bic = -2 * ll * n + n_params * np.log(n)  # hmmlearn score is avg loglik
        # actually model.score returns mean log-likelihood per sample
        bic = -2 * ll * n + n_params * np.log(n)
        dur = mean_state_duration(model.transmat_)
        scores[k] = {"bic": float(bic), "mean_duration": dur, "ll": float(ll)}

    eligible = [k for k, s in scores.items() if s["mean_duration"] >= min_duration]
    pool = eligible if eligible else list(scores.keys())
    best = min(pool, key=lambda k: scores[k]["bic"])
    return best, scores


def fit_filter_fold(
    X_regime: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_components: int,
    random_state: int = 0,
    vol_index: int = 0,
) -> RegimeFitResult:
    """
    Refit HMM on TRAIN only, then filter forward through train+test with frozen params.
    """
    X_train = X_regime[train_idx]
    model = fit_gaussian_hmm(
        X_train, n_components=n_components, random_state=random_state, vol_index=vol_index
    )
    end = int(test_idx[-1]) + 1
    s_full = filtered_posteriors(model, X_regime[:end])
    bic_ll = model.score(X_train)
    n, d = X_train.shape
    n_cov = d * (d + 1) / 2
    n_params = n_components * (d + n_cov) + n_components * (n_components - 1) + (n_components - 1)
    bic = -2 * bic_ll * n + n_params * np.log(n)
    return RegimeFitResult(
        k=n_components,
        bic=float(bic),
        mean_duration=mean_state_duration(model.transmat_),
        model=model,
        filtered=s_full,
    )


def regime_feature_frame(
    index: pd.DatetimeIndex,
    filtered: np.ndarray,
    prefix: str = "p_regime",
) -> pd.DataFrame:
    cols = [f"{prefix}_{i}" for i in range(filtered.shape[1])]
    return pd.DataFrame(filtered, index=index[: len(filtered)], columns=cols)
