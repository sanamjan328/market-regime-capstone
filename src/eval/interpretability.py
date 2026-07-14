"""FiLM gamma/beta analysis, SHAP on LightGBM, self-attention weight extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def extract_film_params(
    model: nn.Module,
    n_regimes: int,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """
    Probe FiLM layers by passing each pure regime one-hot vector e_k and
    recording the resulting (gamma, beta) for every encoder block.

    Columns: regime, block, gamma_mean, gamma_norm, beta_mean, beta_norm.

    If gamma/beta are near-identical across regimes the FiLM layer learned
    nothing and you should report that finding.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    rows = []
    with torch.no_grad():
        for k in range(n_regimes):
            s = torch.zeros(1, n_regimes, device=device)
            s[0, k] = 1.0
            for i, block in enumerate(model.blocks):
                if not hasattr(block, "film") or block.film is None:
                    continue
                params = block.film.net(s).squeeze(0)  # (2 * d_model,)
                d = params.shape[0] // 2
                gamma = params[:d].cpu().numpy()
                beta = params[d:].cpu().numpy()
                rows.append(
                    {
                        "regime": k,
                        "block": i,
                        "gamma_mean": float(gamma.mean()),
                        "gamma_norm": float(np.linalg.norm(gamma)),
                        "beta_mean": float(beta.mean()),
                        "beta_norm": float(np.linalg.norm(beta)),
                    }
                )
    return pd.DataFrame(rows)


def film_regime_divergence(film_params: pd.DataFrame) -> pd.DataFrame:
    """
    For each encoder block compute the mean L2 distance between regime
    gamma vectors.  Near-zero divergence means FiLM is not differentiating
    between regimes -- a diagnostic to report.
    """
    rows = []
    for block, grp in film_params.groupby("block"):
        gamma_norms = grp.set_index("regime")["gamma_norm"].to_dict()
        beta_norms = grp.set_index("regime")["beta_norm"].to_dict()
        regimes = sorted(gamma_norms.keys())
        if len(regimes) < 2:
            continue
        diffs_gamma = []
        diffs_beta = []
        for i in range(len(regimes)):
            for j in range(i + 1, len(regimes)):
                diffs_gamma.append(abs(gamma_norms[regimes[i]] - gamma_norms[regimes[j]]))
                diffs_beta.append(abs(beta_norms[regimes[i]] - beta_norms[regimes[j]]))
        rows.append(
            {
                "block": block,
                "mean_gamma_divergence": float(np.mean(diffs_gamma)),
                "mean_beta_divergence": float(np.mean(diffs_beta)),
            }
        )
    return pd.DataFrame(rows)


def shap_lgbm(
    model,
    X: np.ndarray,
    feature_names: list[str],
    max_display: int = 20,
) -> pd.DataFrame:
    """
    Compute mean |SHAP| per feature for a fitted LightGBM model.
    Falls back gracefully if the shap package is not installed.
    """
    try:
        import shap
    except ImportError:
        return pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": [float("nan")] * len(feature_names),
                "note": ["shap not installed"] * len(feature_names),
            }
        )
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        shap_vals = np.abs(np.array(shap_vals)).mean(axis=0)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    return df.sort_values("mean_abs_shap", ascending=False).head(max_display).reset_index(drop=True)


def attention_weights(
    model: nn.Module,
    x: torch.Tensor,
    s: torch.Tensor,
) -> list[np.ndarray]:
    """
    Extract per-block self-attention weight matrices.

    x : (1, L, F)  --  one lookback window
    s : (1, K)     --  filtered regime posterior
    Returns list of (N_patches, N_patches) arrays, one per encoder block.
    """
    model.eval()
    collected: list[np.ndarray] = []
    with torch.no_grad():
        h = model.patch(x) + model.pos
        for block in model.blocks:
            h_norm = block.norm1(h)
            _, w = block.attn(h_norm, h_norm, h_norm, need_weights=True)
            collected.append(w.squeeze(0).cpu().numpy())  # (N, N)
            attn_out, _ = block.attn(h_norm, h_norm, h_norm, need_weights=False)
            h = h + attn_out
            if block.use_film and s is not None and block.film is not None:
                h = block.film(h, s)
            h = h + block.ff(block.norm2(h))
    return collected
