#!/usr/bin/env python
"""
Day-19 BTC robustness check: re-run ablation on BTC-USD hourly data and
compare regime structure + performance to the SPY primary results.

Prerequisites (run once with data=btc_hourly):
    python scripts/build_features.py data=btc_hourly
    python scripts/fit_regimes.py data=btc_hourly
    python scripts/run_baselines.py data=btc_hourly
    python scripts/train.py model=patchtst_no_regime data=btc_hourly
    python scripts/train.py model=patchtst_film data=btc_hourly
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig


PRED_FILES = {
    "buy_hold": "buy_hold_oos.parquet",
    "arima_garch": "arima_garch_oos.parquet",
    "transformer_no_regime": "patchtst_no_regime_oos.parquet",
    "transformer_film": "patchtst_film_oos.parquet",
}


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    from src.backtest.costs import total_cost
    from src.backtest.metrics import calmar, max_drawdown, sharpe, sortino
    from src.backtest.sizing import kelly_lite_positions
    from src.eval.bootstrap import bootstrap_sharpe_ci

    universe = cfg.data.universe.lower()  # "btc"
    feat_path = Path(cfg.data.processed_dir) / f"{universe}_features_regimes.parquet"
    panel_path = Path(cfg.data.processed_dir) / f"{universe}_panel.parquet"
    pred_dir = Path(cfg.output_dir) / "predictions"
    out_dir = Path(cfg.output_dir)

    if not feat_path.exists():
        print(
            f"BTC features not found at {feat_path}.\n"
            "Generate them first:\n"
            "  python scripts/build_features.py data=btc_hourly\n"
            "  python scripts/fit_regimes.py data=btc_hourly\n"
            "Then retrain models with data=btc_hourly and re-run this script."
        )
        return

    if not panel_path.exists():
        print(f"BTC panel not found at {panel_path}. Run scripts/download_data.py first.")
        return

    feats = pd.read_parquet(feat_path)
    panel = pd.read_parquet(panel_path)
    regime_cols = [c for c in feats.columns if c.startswith("p_regime_")]

    # Annualisation factor: hourly data has ~8760 bars/year
    periods = 8760 if "1h" in cfg.data.get("interval", "1d") else 252

    print(f"BTC robustness check — universe={universe.upper()}  periods/year={periods}")
    print("=" * 60)

    rows = []
    for model_name, fname in PRED_FILES.items():
        p = pred_dir / fname
        if not p.exists():
            print(f"  skip {model_name}: {fname} missing")
            continue

        preds = pd.read_parquet(p)
        idx = preds.index.intersection(feats.index).intersection(panel.index)
        if len(idx) < 100:
            print(f"  skip {model_name}: too few overlapping rows ({len(idx)})")
            continue

        preds_s = preds.loc[idx]
        feats_s = feats.loc[idx]
        regimes = feats_s[regime_cols].values if regime_cols else None
        rv_col = "realized_vol_20d" if "realized_vol_20d" in feats_s.columns else feats_s.columns[0]
        rv = feats_s[rv_col].ffill().fillna(0.01).values

        pos = kelly_lite_positions(
            preds_s["mu"].values,
            preds_s["sigma"].values,
            rv,
            regimes,
            vol_target=float(cfg.backtest.vol_target),
            leverage_cap=float(cfg.backtest.leverage_cap),
        )
        executed = pd.Series(pos, index=idx).shift(1).fillna(0.0)
        bar_ret = panel["close"].reindex(idx).pct_change().fillna(0.0)
        gross_arr = executed.values * bar_ret.values
        costs_arr = total_cost(executed.values, commission_bps=float(cfg.backtest.commission_bps))
        net_rets = pd.Series(gross_arr - costs_arr, index=idx)

        r = net_rets.dropna()
        ci = bootstrap_sharpe_ci(r, n_boot=500, mean_block=max(1, periods // 52), seed=int(cfg.seed))
        mdd, _ = max_drawdown(r)

        rows.append(
            {
                "model": model_name,
                "sharpe": ci["sharpe"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "sortino": sortino(r),
                "calmar": calmar(r),
                "max_dd": mdd,
                "ann_return": float(r.mean() * periods),
                "n_obs": len(r),
            }
        )
        ci_str = f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"
        print(f"  {model_name:<28} Sharpe={ci['sharpe']:+.3f} {ci_str}")

    if not rows:
        print("No BTC prediction files found. Run the pipeline with data=btc_hourly.")
        return

    df = pd.DataFrame(rows)
    out_csv = out_dir / "btc_robustness.csv"
    df.to_csv(out_csv, index=False)

    # Load SPY results for comparison
    spy_abl = out_dir.parent / cfg.output_dir / "ablation_table.csv"
    abl_path = Path(cfg.output_dir) / "ablation_table.csv"
    if abl_path.exists():
        spy_df = pd.read_csv(abl_path)[["model", "sharpe"]].rename(columns={"sharpe": "spy_sharpe"})
        comparison = df[["model", "sharpe"]].rename(columns={"sharpe": "btc_sharpe"}).merge(
            spy_df, on="model", how="left"
        )
        print("\n── SPY vs BTC Sharpe comparison ──")
        print(comparison.to_string(index=False))

    print(f"\nWrote {out_csv}")

    # Regime structure comparison
    if regime_cols:
        print("\n── BTC regime statistics ──")
        hard = feats[regime_cols].values.argmax(axis=1)
        for k, col in enumerate(regime_cols):
            pct = (hard == k).mean()
            print(f"  regime {k}: {pct:.1%} of bars")


if __name__ == "__main__":
    main()
