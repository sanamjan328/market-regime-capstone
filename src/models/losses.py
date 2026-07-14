"""Losses for distributional forecasting."""

from __future__ import annotations

import torch


def gaussian_nll(
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    y: torch.Tensor,
    log_sigma_min: float = -5.0,
    log_sigma_max: float = 2.0,
) -> torch.Tensor:
    log_sigma = log_sigma.clamp(log_sigma_min, log_sigma_max)
    sigma_sq = torch.exp(2.0 * log_sigma)
    return 0.5 * (2.0 * log_sigma + (y - mu) ** 2 / sigma_sq).mean()


def pinball_loss(
    pred_quantiles: torch.Tensor,
    y: torch.Tensor,
    qs: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> torch.Tensor:
    """
    pred_quantiles: (B, Q)
    y: (B,)
    """
    y = y.unsqueeze(-1)
    losses = []
    for i, q in enumerate(qs):
        e = y - pred_quantiles[:, i : i + 1]
        losses.append(torch.maximum(q * e, (q - 1) * e))
    return torch.cat(losses, dim=-1).mean()
