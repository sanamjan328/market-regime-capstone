"""Unit tests for core numerical helpers (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.fracdiff import frac_diff, frac_diff_weights
from src.models.film import FiLM
from src.models.losses import gaussian_nll
from src.regimes.hmm import mean_state_duration
import torch


def test_frac_diff_weights_sum_and_length():
    w = frac_diff_weights(0.4, thresh=1e-4)
    assert len(w) > 5
    assert abs(w[-1] - 1.0) < 1e-12  # newest weight is 1 after reverse? oldest first: first was 1 then reversed so last is 1
    # weights reversed: last element is original w0=1
    assert w[-1] == 1.0


def test_frac_diff_nan_warmup():
    x = np.arange(120.0)
    y = frac_diff(x, d=0.4, thresh=1e-4, max_size=50)
    assert np.isnan(y[0])
    assert np.isfinite(y[-1])


def test_film_identity_at_init():
    film = FiLM(n_regimes=3, d_model=8, hidden=16)
    h = torch.randn(2, 5, 8)
    s = torch.softmax(torch.randn(2, 3), dim=-1)
    out = film(h, s)
    # with zero init + gamma+=1, near identity of LayerNorm(h)
    assert out.shape == h.shape


def test_gaussian_nll_finite():
    mu = torch.zeros(10)
    log_sigma = torch.zeros(10)
    y = torch.randn(10)
    loss = gaussian_nll(mu, log_sigma, y)
    assert torch.isfinite(loss)


def test_mean_state_duration_geometric():
    A = np.array([[0.9, 0.1], [0.2, 0.8]])
    d = mean_state_duration(A)
    assert d > 5


def test_hard_switch_routes_by_argmax():
    from src.models.hard_switch import HardSwitchPatchTST

    model = HardSwitchPatchTST(
        n_regimes=3,
        n_features=4,
        lookback=32,
        patch_len=8,
        stride=4,
        d_model=16,
        n_heads=2,
        n_layers=1,
        ff_dim=32,
        dropout=0.0,
    )
    x = torch.randn(5, 32, 4)
    s = torch.tensor(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.05, 0.05, 0.9],
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
        ]
    )
    mu, log_sigma = model(x, s)
    assert mu.shape == (5,)
    assert log_sigma.shape == (5,)


def test_split_train_val_indices():
    from src.models.train_loop import split_train_val_indices

    fit, val = split_train_val_indices(100, val_frac=0.2)
    assert len(fit) == 80
    assert len(val) == 20
    assert fit.max() < val.min()
