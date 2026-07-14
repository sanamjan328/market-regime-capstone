"""Shared training helpers: val-Sharpe early stop, prediction helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.backtest.metrics import sharpe
from src.models.losses import gaussian_nll, pinball_loss


@dataclass
class TrainResult:
    best_state: dict
    best_metric: float
    epochs_run: int


def strategy_sharpe(mu: np.ndarray, y: np.ndarray) -> float:
    """Proxy Sharpe used for early stopping (position = clip(mu))."""
    pos = np.clip(mu, -1.0, 1.0)
    rets = pos * y
    return sharpe(pd.Series(rets))


def split_train_val_indices(n: int, val_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Chronological split of dataset indices (last val_frac -> validation)."""
    n_val = max(1, int(round(n * val_frac)))
    n_fit = max(1, n - n_val)
    if n_fit + n_val > n:
        n_val = n - n_fit
    fit_idx = np.arange(0, n_fit)
    val_idx = np.arange(n_fit, n)
    if len(val_idx) == 0:
        val_idx = fit_idx[-1:]
    return fit_idx, val_idx


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_s: bool,
    loss_type: str = "gaussian",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    mus, sigs, ys = [], [], []
    for xb, yb, sb in loader:
        xb, sb = xb.to(device), sb.to(device)
        out = model(xb, sb if use_s else None)
        if loss_type == "pinball":
            q = out  # (B, Q)
            mu = q[:, 1]
            sigma = ((q[:, 2] - q[:, 0]).abs() / 2.0).clamp_min(1e-6)
        else:
            mu, log_sigma = out
            sigma = torch.exp(log_sigma.clamp(-5.0, 2.0))
        mus.append(mu.cpu().numpy())
        sigs.append(sigma.cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(mus), np.concatenate(sigs), np.concatenate(ys)


def train_one(
    model: nn.Module,
    train_ds,
    device: torch.device,
    *,
    use_s: bool,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    max_epochs: int = 50,
    patience: int = 8,
    val_frac: float = 0.2,
    loss_type: str = "gaussian",
    quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> TrainResult:
    """
    Train with early stopping on validation Sharpe (not train NLL/MSE).
    """
    fit_idx, val_idx = split_train_val_indices(len(train_ds), val_frac=val_frac)
    fit_loader = DataLoader(Subset(train_ds, fit_idx.tolist()), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(train_ds, val_idx.tolist()), batch_size=256, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state, best_sharpe, wait = None, -np.inf, 0
    epochs_run = 0

    for epoch in range(max_epochs):
        epochs_run = epoch + 1
        model.train()
        for xb, yb, sb in fit_loader:
            xb, yb, sb = xb.to(device), yb.to(device), sb.to(device)
            opt.zero_grad()
            out = model(xb, sb if use_s else None)
            if loss_type == "pinball":
                loss = pinball_loss(out, yb, qs=quantile_levels)
            else:
                mu, log_sigma = out
                loss = gaussian_nll(mu, log_sigma, yb)
            loss.backward()
            opt.step()

        mu_val, _, y_val = predict_loader(model, val_loader, device, use_s, loss_type=loss_type)
        val_s = strategy_sharpe(mu_val, y_val)
        if val_s > best_sharpe:
            best_sharpe = val_s
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_sharpe = float("-inf")
    return TrainResult(best_state=best_state, best_metric=float(best_sharpe), epochs_run=epochs_run)
