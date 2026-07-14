"""Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def probabilistic_sharpe_ratio(
    sharpe: float,
    sharpe_benchmark: float,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """PSR: Prob[observed SR > SR*] under non-normal returns."""
    if n_obs <= 1:
        return 0.0
    numer = (sharpe - sharpe_benchmark) * np.sqrt(n_obs - 1)
    denom = np.sqrt(1 - skew * sharpe + ((kurt - 1) / 4.0) * sharpe**2)
    if denom <= 0:
        return 0.0
    return float(norm.cdf(numer / denom))


def expected_max_sharpe(n_trials: int, sharpe_var: float = 1.0) -> float:
    """E[max SR] under n independent trials ~ N(0, sharpe_var)."""
    if n_trials <= 1:
        return 0.0
    # Euler-Mascheroni approximation
    gamma = 0.5772156649
    return float(np.sqrt(sharpe_var) * ((1 - gamma) * norm.ppf(1 - 1 / n_trials) + gamma * norm.ppf(1 - 1 / (n_trials * np.e))))


def deflated_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    n_trials: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """
    DSR = PSR(SR*, SR̂) where SR* is the expected max SR under multiple testing.
    """
    sr_star = expected_max_sharpe(n_trials)
    return probabilistic_sharpe_ratio(sharpe, sr_star, n_obs, skew=skew, kurt=kurt)
