"""Causal filtered HMM posteriors via the forward algorithm.

CRITICAL: hmmlearn's predict_proba() returns SMOOTHED posteriors
P(s_t | x_1..x_T) which leak future information. Use this module instead.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def filtered_posteriors(model, X: np.ndarray) -> np.ndarray:
    """
    Causal P(s_t | x_1..x_t) via the forward algorithm.

    Parameters
    ----------
    model : fitted hmmlearn.GaussianHMM
    X : (T, d) feature matrix

    Returns
    -------
    (T, K) array of filtered state probabilities.
    """
    framelogprob = model._compute_log_likelihood(X)  # (T, K)
    T, K = framelogprob.shape
    log_alpha = np.zeros((T, K))
    log_A = np.log(model.transmat_ + 1e-300)
    log_pi = np.log(model.startprob_ + 1e-300)

    log_alpha[0] = log_pi + framelogprob[0]
    for t in range(1, T):
        log_alpha[t] = (
            logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0) + framelogprob[t]
        )
    return np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))


def canonicalize_states(model, vol_index: int = 0) -> np.ndarray:
    """
    Sort HMM states by ascending mean of the volatility feature (vol_index).

    EM does not preserve state ordering across refits — regime 0 in fold 3 may
    be regime 2 in fold 4. Canonicalise after every fit.
    Returns the permutation applied.
    """
    means = np.asarray(model.means_)
    order = np.argsort(means[:, vol_index])
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))

    model.means_ = means[order]
    if hasattr(model, "covars_"):
        model.covars_ = np.asarray(model.covars_)[order]
    model.startprob_ = np.asarray(model.startprob_)[order]
    # permute transition rows and columns
    A = np.asarray(model.transmat_)
    model.transmat_ = A[order][:, order]
    return order
