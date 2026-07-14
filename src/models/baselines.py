"""Classical / ML baselines for the ablation table."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BaselinePrediction:
    mu: np.ndarray
    sigma: np.ndarray
    name: str


def buy_and_hold(n: int, realized_vol: np.ndarray) -> BaselinePrediction:
    """Always-long benchmark (position proxy mu=1)."""
    return BaselinePrediction(mu=np.ones(n), sigma=np.asarray(realized_vol, dtype=float), name="buy_hold")


def zero_forecast(n: int, realized_vol: np.ndarray) -> BaselinePrediction:
    return BaselinePrediction(mu=np.zeros(n), sigma=np.asarray(realized_vol, dtype=float), name="zero")


def fit_arima_garch(
    train_returns: pd.Series,
    test_returns: pd.Series,
    realized_vol_test: np.ndarray | None = None,
    refit_every: int = 20,
) -> BaselinePrediction:
    """
    ARIMA(1)-GARCH(1,1) with rolling one-step forecasts.

    Refits every `refit_every` test steps to keep runtime tractable while still
    producing a time-varying path (not a single constant forecast).
    """
    try:
        from arch import arch_model
    except ImportError as exc:  # pragma: no cover
        raise ImportError("arch package required for ARIMA-GARCH baseline") from exc

    hist = (train_returns.dropna() * 100.0).astype(float)
    test = (test_returns.fillna(0.0) * 100.0).astype(float)
    mus: list[float] = []
    sigs: list[float] = []
    res = None

    for i in range(len(test)):
        if res is None or i % refit_every == 0:
            am = arch_model(
                hist.values,
                mean="AR",
                lags=1,
                vol="Garch",
                p=1,
                q=1,
                dist="normal",
                rescale=False,
            )
            res = am.fit(disp="off", show_warning=False)
        fcast = res.forecast(horizon=1, reindex=False)
        mu_pct = float(fcast.mean.values[-1, 0])
        var_pct = float(fcast.variance.values[-1, 0])
        mus.append(mu_pct / 100.0)
        sigs.append(max(np.sqrt(max(var_pct, 1e-12)) / 100.0, 1e-6))
        # expand history with realised test return (causal: available after bar i)
        hist = pd.concat([hist, test.iloc[i : i + 1]])

    sigma = np.asarray(sigs, dtype=float)
    if realized_vol_test is not None and len(realized_vol_test) == len(sigma):
        # blend modeled vol with realised vol floor for stability
        sigma = np.maximum(sigma, np.asarray(realized_vol_test, dtype=float) * 0.5)
    return BaselinePrediction(mu=np.asarray(mus, dtype=float), sigma=sigma, name="arima_garch")


def fit_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    realized_vol_test: np.ndarray,
) -> BaselinePrediction:
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover
        raise ImportError("lightgbm required") from exc

    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )
    mask = np.isfinite(y_train) & np.all(np.isfinite(X_train), axis=1)
    model.fit(X_train[mask], y_train[mask])
    mu = model.predict(np.nan_to_num(X_test))
    return BaselinePrediction(mu=mu, sigma=np.asarray(realized_vol_test, dtype=float), name="lightgbm")
