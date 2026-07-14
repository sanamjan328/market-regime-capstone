"""PatchTST backbone with optional FiLM / cross-attention regime conditioning."""

from __future__ import annotations

import torch
import torch.nn as nn

from .film import FiLM


class PatchEmbedding(nn.Module):
    def __init__(self, n_features: int, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        # channel-independent: embed each feature's patches separately then avg/sum
        self.proj = nn.Linear(patch_len, d_model)
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, F)
        returns: (B, N_patches, d_model) — mean over feature channels
        """
        B, L, F = x.shape
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)  # (B, N, F, P)
        B, N, F, P = patches.shape
        patches = patches.permute(0, 2, 1, 3).reshape(B * F, N, P)
        emb = self.proj(patches)  # (B*F, N, d)
        emb = emb.reshape(B, F, N, -1).mean(dim=1)  # channel independence via mean pool
        return emb


class RegimeCrossAttention(nn.Module):
    """Cross-attention from patch tokens to a learned K x d_model regime table."""

    def __init__(self, n_regimes: int, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.table = nn.Parameter(torch.randn(n_regimes, d_model) * 0.02)
        self.cross = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # Soft-weighted regime embedding as a single KV token, plus full table.
        soft = (s @ self.table).unsqueeze(1)  # (B, 1, d)
        table = self.table.unsqueeze(0).expand(h.size(0), -1, -1)  # (B, K, d)
        kv = torch.cat([soft, table], dim=1)
        h_n = self.norm(h)
        out, _ = self.cross(h_n, kv, kv, need_weights=False)
        return h + out


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        dropout: float,
        n_regimes: int,
        film_hidden: int,
        use_film: bool,
        use_cross_attn: bool,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.use_film = use_film
        self.use_cross_attn = use_cross_attn
        self.film = FiLM(n_regimes, d_model, film_hidden) if use_film else None
        self.cross = (
            RegimeCrossAttention(n_regimes, d_model, n_heads, dropout) if use_cross_attn else None
        )

    def forward(self, h: torch.Tensor, s: torch.Tensor | None) -> torch.Tensor:
        h_norm = self.norm1(h)
        attn_out, _ = self.attn(h_norm, h_norm, h_norm, need_weights=False)
        h = h + attn_out
        if self.use_film and s is not None:
            h = self.film(h, s)
        if self.use_cross_attn and s is not None:
            h = self.cross(h, s)
        h = h + self.ff(self.norm2(h))
        return h


class PatchTST(nn.Module):
    def __init__(
        self,
        n_features: int,
        lookback: int = 60,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.2,
        ff_dim: int = 256,
        n_regimes: int = 3,
        film_hidden: int = 64,
        use_regime: bool = True,
        conditioning: str = "film",
        loss_type: str = "gaussian",
        n_quantiles: int = 3,
    ):
        super().__init__()
        self.use_regime = use_regime and conditioning not in ("none",)
        self.conditioning = conditioning
        self.loss_type = loss_type
        self.patch = PatchEmbedding(n_features, patch_len, stride, d_model)
        n_patches = 1 + (lookback - patch_len) // stride
        self.pos = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.normal_(self.pos, std=0.02)

        use_film = self.use_regime and conditioning == "film"
        use_cross = self.use_regime and conditioning == "cross_attn"
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    d_model,
                    n_heads,
                    ff_dim,
                    dropout,
                    n_regimes,
                    film_hidden,
                    use_film,
                    use_cross,
                )
                for _ in range(n_layers)
            ]
        )
        self.concat_proj = None
        if self.use_regime and conditioning == "concat":
            self.concat_proj = nn.Linear(d_model + n_regimes, d_model)

        out_dim = n_quantiles if loss_type == "pinball" else 2
        self.head = nn.Linear(d_model, out_dim)
        self.n_regimes = n_regimes

    def forward(self, x: torch.Tensor, s: torch.Tensor | None = None):
        """
        x: (B, L, F)
        s: (B, K) filtered posterior
        returns:
          gaussian: (mu, log_sigma) each (B,)
          pinball:  (B, Q) quantile predictions
        """
        h = self.patch(x) + self.pos
        if self.use_regime and self.conditioning == "concat" and s is not None:
            s_exp = s.unsqueeze(1).expand(-1, h.size(1), -1)
            h = self.concat_proj(torch.cat([h, s_exp], dim=-1))

        regime_s = s if self.conditioning in ("film", "cross_attn") else None
        for block in self.blocks:
            h = block(h, regime_s)

        pooled = h.mean(dim=1)
        out = self.head(pooled)
        if self.loss_type == "pinball":
            return out
        mu, log_sigma = out[:, 0], out[:, 1]
        return mu, log_sigma
