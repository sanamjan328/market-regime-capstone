"""Hard regime switching: K separate Transformers, dispatch by argmax posterior."""

from __future__ import annotations

import torch
import torch.nn as nn

from .patchtst import PatchTST


class HardSwitchPatchTST(nn.Module):
    """
    Model #7 in the ablation table.

    Trains K unconditional PatchTST experts. At inference (and in the forward
    for loss), each sample is routed to the expert matching argmax(s_t).
    """

    def __init__(self, n_regimes: int = 3, **patchtst_kwargs):
        super().__init__()
        kwargs = dict(patchtst_kwargs)
        kwargs["use_regime"] = False
        kwargs["conditioning"] = "none"
        kwargs["n_regimes"] = n_regimes
        kwargs["loss_type"] = "gaussian"  # hard-switch ablation uses Gaussian NLL
        self.n_regimes = n_regimes
        self.experts = nn.ModuleList([PatchTST(**kwargs) for _ in range(n_regimes)])

    def forward(self, x: torch.Tensor, s: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, L, F)
        s: (B, K) filtered posterior — required for routing
        """
        if s is None:
            raise ValueError("HardSwitchPatchTST requires regime posteriors s for routing")
        hard = s.argmax(dim=-1)  # (B,)
        mus: list[torch.Tensor] = []
        log_sigmas: list[torch.Tensor] = []
        for expert in self.experts:
            mu_k, ls_k = expert(x, None)
            mus.append(mu_k)
            log_sigmas.append(ls_k)
        mu_stack = torch.stack(mus, dim=1)  # (B, K)
        ls_stack = torch.stack(log_sigmas, dim=1)
        gather_idx = hard.unsqueeze(1)
        mu = mu_stack.gather(1, gather_idx).squeeze(1)
        log_sigma = ls_stack.gather(1, gather_idx).squeeze(1)
        return mu, log_sigma
