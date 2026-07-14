#!/usr/bin/env python
"""
Day-18 sensitivity analysis: vary transaction costs (0/5/10/20 bps) across all
ablation models and report whether the edge survives at 10 bps.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

COST_GRID_BPS = [0, 5, 10, 20]

PRED_FILES = {
    "buy_hold": "buy_hold_oos.parquet",
    "arima_garch": "arima_garch_oos.parquet",
    "dlinear": "dlinear_oos.parquet",
    "lightgbm": "lightgbm_oos.parquet",
    "transformer_no_regime": "patchtst_no_regime_oos.parquet",
    "transformer_film": "patchtst_film_oos.parquet",
    "transformer_hard_switch": "patchtst_hard_switch_oos.parquet",
}


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    from src.backtest.costs import total_cost
    from src.backtest.metrics import sharpe
    from src.backtest.sizing import kelly_lite_positions

    feat_path = Path(cfg.data.processed_dir) / f"{cfg.data.universe.lower()}_features_regimes.parquet"
    panel_path = Path(cfg.data.processed_dir) / f"{cfg.data.universe.lower()}_panel.parquet"
    pred_dir = Path(cfg.output_dir) / "predictions"

    feats = pd.read_parquet(feat_path)
    panel = pd.read_parquet(panel_path)
    regime_cols = [c for c in feats.columns if c.startswith("p_regime_")]

    rows = []
    for model_name, fname in PRED_FILES.items():
        p = pred_dir / fname
        if not p.exists():
            print(f"  skip {model_name}: {fname} missing")
            continue

        preds = pd.read_parquet(p)
        idx = preds.index.intersection(feats.index).intersection(panel.index)
        preds_sub = preds.loc[idx]
        feats_sub = feats.loc[idx]
        regimes = feats_sub[regime_cols].values if regime_cols else None
        rv = feats_sub["realized_vol_20d"].ffill().fillna(0.01).values

        pos = kelly_lite_positions(
            preds_sub["mu"].values,
            preds_sub["sigma"].values,
            rv,
            regimes,
            vol_target=float(cfg.backtest.vol_target),
            leverage_cap=float(cfg.backtest.leverage_cap),
        )
        executed = pd.Series(pos, index=idx).shift(1).fillna(0.0)
        bar_ret = panel["close"].reindex(idx).pct_change().fillna(0.0)
        gross = (executed.values * bar_ret.values)

        for bps in COST_GRID_BPS:
            costs = total_cost(executed.values, commission_bps=float(bps), slippage_k=0.0)
            net = gross - costs
            rows.append(
                {
                    "model": model_name,
                    "commission_bps": bps,
                    "sharpe": sharpe(pd.Series(net)),
                    "ann_return": float(net.mean() * 252),
                }
            )

        print(f"  {model_name}: done")

    df = pd.DataFrame(rows)

    # Pivot to wide: one column per cost level
    sharpe_pivot = df.pivot(index="model", columns="commission_bps", values="sharpe")
    sharpe_pivot.columns = [f"sharpe_{c}bps" for c in sharpe_pivot.columns]
    sharpe_pivot["edge_survives_10bps"] = sharpe_pivot.get("sharpe_10bps", pd.Series(dtype=float)) > 0
    sharpe_pivot = sharpe_pivot.reset_index()

    out = Path(cfg.output_dir) / "sensitivity_costs.csv"
    sharpe_pivot.to_csv(out, index=False)

    print("\n── Cost sensitivity (Sharpe by commission level) ──")
    print(sharpe_pivot.to_string(index=False))
    print(f"\nWrote {out}")

    # Flag models whose edge dies before 10 bps
    dead = sharpe_pivot[~sharpe_pivot.get("edge_survives_10bps", True)]["model"].tolist()
    if dead:
        print(f"\nWARNING: Edge dies below 10 bps for: {dead}")
        print("  This means the signal is real but the strategy is too expensive to trade.")
    else:
        print("\nAll models retain positive Sharpe through 10 bps.")


if __name__ == "__main__":
    main()
