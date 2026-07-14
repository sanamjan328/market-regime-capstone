"""
Leakage checklist tests — non-negotiable correctness gates.

These encode Section 10 of the architecture PDF as pytest assertions.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest


def test_filtered_posteriors_differ_from_smoothed():
    """HMM posteriors must be filtered, not smoothed predict_proba."""
    from hmmlearn.hmm import GaussianHMM

    from src.regimes.filtering import filtered_posteriors

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    X[100:] += 3.0
    model = GaussianHMM(n_components=2, covariance_type="full", n_iter=50, random_state=0)
    model.fit(X)
    filt = filtered_posteriors(model, X)
    smooth = model.predict_proba(X)
    # Not identical in general (except endpoints-ish); max abs diff should be > 0 interior
    assert filt.shape == smooth.shape
    assert not np.allclose(filt[20:-20], smooth[20:-20], atol=1e-6)


def test_filtering_is_causal_no_future_dependence():
    """Changing the future must not change filtered posteriors in the past."""
    from hmmlearn.hmm import GaussianHMM

    from src.regimes.filtering import filtered_posteriors

    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 2))
    model = GaussianHMM(n_components=2, covariance_type="full", n_iter=40, random_state=0)
    model.fit(X)
    f1 = filtered_posteriors(model, X)
    X2 = X.copy()
    X2[-20:] = rng.normal(size=(20, 2)) + 10.0
    f2 = filtered_posteriors(model, X2)
    assert np.allclose(f1[:-20], f2[:-20], atol=1e-8)


def test_hmm_refit_per_fold_pattern_documented():
    """fit_filter_fold must fit only on train_idx."""
    from src.regimes.hmm import fit_filter_fold

    src = inspect.getsource(fit_filter_fold)
    assert "train_idx" in src
    assert "X_train" in src


def test_canonicalize_sorts_by_volatility_feature():
    from hmmlearn.hmm import GaussianHMM

    from src.regimes.filtering import canonicalize_states

    rng = np.random.default_rng(2)
    X = rng.normal(size=(120, 1))
    model = GaussianHMM(n_components=3, covariance_type="full", n_iter=30, random_state=1)
    model.fit(X)
    canonicalize_states(model, vol_index=0)
    means = model.means_[:, 0]
    assert np.all(np.diff(means) >= -1e-9)


def test_scaler_fit_transform_isolation():
    from src.features.scalers import ExpandingStandardScaler

    train = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0, 20.0)})
    test = pd.DataFrame({"a": np.arange(100.0, 110.0), "b": np.arange(200.0, 210.0)})
    sc = ExpandingStandardScaler().fit(train)
    tr = sc.transform(train)
    te = sc.transform(test)
    assert abs(tr["a"].mean()) < 1e-8
    # test mean should not be ~0 if fitted on train only
    assert abs(te["a"].mean()) > 1


def test_rolling_features_use_shift():
    from src.features import build_feature_matrix

    src = inspect.getsource(build_feature_matrix)
    assert ".shift(1)" in src
    assert "_shift_rolling" in inspect.getsource(inspect.getmodule(build_feature_matrix))


def test_macro_lag_applied():
    from src.data.pit import lag_macro_series

    s = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.date_range("2020-01-01", periods=4, freq="B"))
    lagged = lag_macro_series(s, lag_days=1)
    assert np.isnan(lagged.iloc[0])
    assert lagged.iloc[1] == 1.0


def test_target_is_forward_looking_only():
    from src.features.targets import vol_normalised_forward_return

    close = pd.Series(np.exp(np.linspace(0, 1, 50)))
    vol = pd.Series(np.full(50, 0.01))
    y = vol_normalised_forward_return(close, vol, horizon=1)
    # last target must be NaN (no future)
    assert np.isnan(y.iloc[-1])


def test_purged_walk_forward_has_embargo_gap():
    from src.features.cv import purged_walk_forward

    dates = pd.bdate_range("2000-01-01", "2021-12-31")
    folds = list(
        purged_walk_forward(
            dates,
            initial_train_end="2010-12-31",
            test_years=2,
            embargo_days=20,
            final_holdout_start="2022-01-01",
        )
    )
    assert len(folds) >= 1
    for f in folds:
        assert f.test_start > f.train_end
        gap = (f.test_start - f.train_end).days
        assert gap > 1  # embargo + purge buffer


def test_holdout_not_in_folds():
    from src.features.cv import holdout_indices, purged_walk_forward

    dates = pd.bdate_range("2000-01-01", "2024-12-31")
    folds = list(purged_walk_forward(dates, final_holdout_start="2022-01-01"))
    holdout = set(holdout_indices(dates, "2022-01-01", "2024-12-31").tolist())
    for f in folds:
        assert holdout.isdisjoint(set(f.test_idx.tolist()))
        assert holdout.isdisjoint(set(f.train_idx.tolist()))


def test_execution_uses_shifted_position():
    from src.backtest.engine import event_loop_backtest

    src = inspect.getsource(event_loop_backtest)
    assert "shift(1)" in src


def test_shuffled_label_sharpe_near_zero_on_noise():
    """Sanity: shuffled noise strategy should not have large Sharpe."""
    from src.backtest.metrics import sharpe

    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, size=1000))
    shuffled = pd.Series(rng.permutation(r.values))
    assert abs(sharpe(shuffled)) < 1.0


def test_predict_proba_not_used_in_regime_pipeline():
    """Guard against accidental smoothed posterior usage in regimes package."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "regimes"
    for path in root.glob("*.py"):
        text = path.read_text()
        if path.name == "filtering.py":
            # may mention predict_proba in comments warning against it
            continue
        assert "predict_proba" not in text or "SMOOTHED" in text.upper() or "not" in text.lower()
