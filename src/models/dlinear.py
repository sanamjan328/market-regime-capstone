"""DLinear baseline (Zeng et al. 2023) — must beat this for a Transformer result."""

from __future__ import annotations

import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F)
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        x_t = self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)
        return x_t


class DLinear(nn.Module):
    def __init__(self, lookback: int, n_features: int, kernel_size: int = 25):
        super().__init__()
        self.ma = MovingAvg(kernel_size)
        self.linear_seasonal = nn.Linear(lookback, 1)
        self.linear_trend = nn.Linear(lookback, 1)
        self.head = nn.Linear(n_features, 2)

    def forward(self, x: torch.Tensor, s: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L, F)
        trend = self.ma(x)
        seasonal = x - trend
        # channel-independent then average: apply linear over time per feature
        seasonal_out = self.linear_seasonal(seasonal.permute(0, 2, 1)).squeeze(-1)  # (B, F)
        trend_out = self.linear_trend(trend.permute(0, 2, 1)).squeeze(-1)
        y = seasonal_out + trend_out
        out = self.head(y)
        return out[:, 0], out[:, 1]
