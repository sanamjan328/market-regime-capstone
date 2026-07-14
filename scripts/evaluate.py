#!/usr/bin/env python
"""
Day-17/18/19 evaluation: full ablation table, bootstrap CI, DSR,
shuffled-label sanity, per-regime P&L attribution, FiLM interpretability.
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig


PRED_MAP = {
    "buy_hold": "buy_hold",
    "zero": "zero",
    "arima_garch": "arima_garch",
    "dlinear": "dlinear",
    "lightgbm": "lightgbm",
    "patchtst_no_regime": "transformer_no_regime",
    "patchtst_film": "transformer_film",
    "patchtst_hard_switch": "transformer_hard_switch",
}


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    from src.eval.ablation import ABLATION_SPEC, build_ablation_table
    from src.eval.bootstrap import bootstrap_sharpe_ci, monte_carlo_drawdown
    from src.eval.dsr import deflated_sharpe_ratio
    from src.eval.interpretability import extract_film_params, film_regime_divergence
    from src.eval.regime_attribution import per_regime_attribution

    out_dir = Path(cfg.output_dir)
    pred_dir = out_dir / "predictions"
    feat_path = Path(cfg.data.processed_dir) / f"{cfg.data.universe.lower()}_features_regimes.parquet"

    # ── 1. Full ablation table (Day 17) ──────────────────────────────────────
    predictions: dict[str, pd.DataFrame] = {}
    for stem, name in PRED_MAP.items():
        p = pred_dir / f"{stem}_oos.parquet"
        if p.exists():
            predictions[name] = pd.read_parquet(p)

    if predictions:
        table = build_ablation_table(predictions)
        order = [m for m, _ in ABLATION_SPEC]
        table["model"] = pd.Categorical(table["model"], categories=order, ordered=True)
        table = table.sort_values("model").reset_index(drop=True)
        abl_path = out_dir / "ablation_table.csv"
        table.to_csv(abl_path, index=False)
        print("── Ablation table ──")
        print(table[["model", "sharpe", "sortino", "calmar", "max_dd", "hit_rate", "ann_return"]].to_string(index=False))
        print(f"Wrote {abl_path}\n")

    # ── 2. Bootstrap CI + DSR + Monte Carlo DD (Day 18) ──────────────────────
    ret_path = out_dir / "backtest_returns.parquet"
    if ret_path.exists():
        rets = pd.read_parquet(ret_path)["net"]
        ci = bootstrap_sharpe_ci(rets, n_boot=1000, mean_block=10, seed=int(cfg.seed))
        skew = float(rets.skew())
        kurt = float(rets.kurtosis() + 3)
        n_trials = len(PRED_MAP) * int(cfg.get("n_seeds", 5))
        dsr = deflated_sharpe_ratio(ci["sharpe"], n_obs=len(rets), n_trials=n_trials, skew=skew, kurt=kurt)
        mc = monte_carlo_drawdown(rets, n_sims=1000, seed=int(cfg.seed))

        # Shuffled-label sanity check: Sharpe should collapse to ~0
        rng = np.random.default_rng(int(cfg.seed))
        shuffled = pd.Series(rng.permutation(rets.values), index=rets.index)
        shuffled_sharpe = bootstrap_sharpe_ci(shuffled, n_boot=200, seed=int(cfg.seed))["sharpe"]

        print("── Bootstrap CI + DSR (Day 18-19) ──")
        print(f"  Sharpe: {ci['sharpe']:.3f}  95% CI [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]")
        print(f"  DSR:    {dsr:.3f}  (n_trials={n_trials})")
        print(f"  MC max-DD percentile: {mc['percentile']:.3f}")
        print(f"  Shuffled-label Sharpe: {shuffled_sharpe:.3f}  {'OK' if abs(shuffled_sharpe) < 0.5 else 'WARNING: possible leakage'}\n")

        stat_summary = {
            **ci,
            "deflated_sharpe": dsr,
            "skew": skew,
            "kurt": kurt,
            "n_trials": n_trials,
            "mc_realised_max_dd": mc["realised_max_dd"],
            "mc_null_mean_dd": mc["null_mean"],
            "mc_dd_percentile": mc["percentile"],
            "shuffled_label_sharpe": shuffled_sharpe,
            "shuffled_ok": bool(abs(shuffled_sharpe) < 0.5),
        }
        stat_path = out_dir / "eval_summary.json"
        with open(stat_path, "w") as f:
            json.dump(stat_summary, f, indent=2)
        print(f"Wrote {stat_path}")
    else:
        print(f"No backtest_returns.parquet found at {ret_path}. Run scripts/backtest.py first.\n")

    # ── 3. Per-regime P&L attribution (Day 17) ───────────────────────────────
    film_preds_path = pred_dir / "patchtst_film_oos.parquet"
    if film_preds_path.exists() and feat_path.exists():
        feats = pd.read_parquet(feat_path)
        regime_cols = [c for c in feats.columns if c.startswith("p_regime_")]
        if regime_cols and ret_path.exists():
            rets = pd.read_parquet(ret_path)["net"]
            regime_df = feats[regime_cols]
            attr = per_regime_attribution(rets, regime_df)
            attr_path = out_dir / "regime_attribution.csv"
            attr.to_csv(attr_path, index=False)
            print("\n── Per-regime P&L attribution ──")
            print(attr[["name", "pct_time", "ann_return", "sharpe", "max_dd"]].to_string(index=False))
            print(f"Wrote {attr_path}")

    # ── 4. FiLM gamma/beta interpretability (Day 19) ─────────────────────────
    ckpt_dir = pred_dir
    film_ckpts = sorted(ckpt_dir.glob("patchtst_film_seed42_fold*.pt"))
    if film_ckpts and feat_path.exists():
        try:
            from src.models.patchtst import PatchTST

            feats = pd.read_parquet(feat_path)
            regime_cols = [c for c in feats.columns if c.startswith("p_regime_")]
            n_regimes = len(regime_cols) if regime_cols else int(cfg.model.get("n_regimes", 3))
            drop = {"y_vol_norm", "y_triple_barrier", "fwd_ret_1", "open_next", *regime_cols}
            feature_cols = [c for c in feats.columns if c not in drop]
            n_features = len(feature_cols)

            device = torch.device("cpu")
            model = PatchTST(
                n_features=n_features,
                lookback=int(cfg.model.get("lookback", 60)),
                patch_len=int(cfg.model.get("patch_len", 16)),
                stride=int(cfg.model.get("stride", 8)),
                d_model=int(cfg.model.get("d_model", 128)),
                n_heads=int(cfg.model.get("n_heads", 4)),
                n_layers=int(cfg.model.get("n_layers", 3)),
                dropout=float(cfg.model.get("dropout", 0.2)),
                ff_dim=int(cfg.model.get("ff_dim", 256)),
                n_regimes=n_regimes,
                film_hidden=int(cfg.model.get("film_hidden", 64)),
                use_regime=True,
                conditioning="film",
            )
            # Use last fold checkpoint (most data seen in training)
            ckpt = film_ckpts[-1]
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))

            film_df = extract_film_params(model, n_regimes=n_regimes, device=device)
            div_df = film_regime_divergence(film_df)

            film_path = out_dir / "film_params.csv"
            div_path = out_dir / "film_divergence.csv"
            film_df.to_csv(film_path, index=False)
            div_df.to_csv(div_path, index=False)

            print("\n── FiLM gamma/beta per regime (last fold checkpoint) ──")
            print(film_df.to_string(index=False))
            print("\n── FiLM regime divergence (near-zero = FiLM not differentiating) ──")
            print(div_df.to_string(index=False))
            print(f"Wrote {film_path}, {div_path}")

            if not div_df.empty:
                mean_div = div_df["mean_gamma_divergence"].mean()
                if mean_div < 0.1:
                    print("\n  WARNING: Low FiLM divergence ({:.4f}). Regime conditioning "
                          "may not be learning regime-specific representations. "
                          "Report this as a finding.".format(mean_div))
        except Exception as e:
            print(f"FiLM interpretability skipped: {e}")


if __name__ == "__main__":
    main()
