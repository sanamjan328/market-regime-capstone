"""Feature-wise Linear Modulation conditioned on the regime posterior."""

from __future__ import annotations

import torch
import torch.nn as nn


class FiLM(nn.Module):
    """Feature-wise Linear Modulation conditioned on the regime posterior."""

    def __init__(self, n_regimes: int, d_model: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_regimes, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        # Initialise final layer to zeros so gamma=0, beta=0 at start;
        # we then add 1 to gamma → identity map for stable early training.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        h : (B, N, d_model) hidden states
        s : (B, K) filtered regime posterior at time t
        """
        gamma, beta = self.net(s).chunk(2, dim=-1)  # each (B, d_model)
        gamma = gamma.unsqueeze(1) + 1.0
        beta = beta.unsqueeze(1)
        return gamma * self.norm(h) + beta
