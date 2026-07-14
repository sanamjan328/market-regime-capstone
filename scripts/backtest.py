#!/usr/bin/env python
"""Run cost-aware backtest from model predictions."""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    from src.backtest.engine import cross_check_engines, event_loop_backtest
    from src.backtest.sizing import kelly_lite_positions

    pred_path = Path(cfg.output_dir) / "predictions" / f"{cfg.model.get('name', 'model')}_oos.parquet"
    feat_path = Path(cfg.data.processed_dir) / f"{cfg.data.universe.lower()}_features_regimes.parquet"
    if not feat_path.exists():
        feat_path = Path(cfg.data.processed_dir) / f"{cfg.data.universe.lower()}_features.parquet"
    panel_path = Path(cfg.data.processed_dir) / f"{cfg.data.universe.lower()}_panel.parquet"

    preds = pd.read_parquet(pred_path)
    feats = pd.read_parquet(feat_path)
    panel = pd.read_parquet(panel_path)

    idx = preds.index.intersection(feats.index).intersection(panel.index)
    preds, feats, panel = preds.loc[idx], feats.loc[idx], panel.loc[idx]

    regime_cols = [c for c in feats.columns if c.startswith("p_regime_")]
    regimes = feats[regime_cols].values if regime_cols else None
    gates = OmegaConf.to_container(cfg.backtest.regime_gates, resolve=True)
    gates = {int(k): float(v) for k, v in gates.items()}

    pos = kelly_lite_positions(
        preds["mu"].values,
        preds["sigma"].values,
        feats["realized_vol_20d"].reindex(idx).ffill().values,
        regimes,
        regime_gates=gates,
        vol_target=float(cfg.backtest.vol_target),
        leverage_cap=float(cfg.backtest.leverage_cap),
    )
    positions = pd.Series(pos, index=idx, name="position")

    result = event_loop_backtest(
        open_prices=panel["open"].reindex(idx),
        close_prices=panel["close"].reindex(idx),
        positions=positions,
        volume=panel["volume"].reindex(idx),
        commission_bps=float(cfg.backtest.commission_bps),
        slippage_k=float(cfg.backtest.slippage_k),
        borrow_bps=float(cfg.backtest.borrow_bps),
    )
    check = cross_check_engines(panel["close"].reindex(idx), positions, float(cfg.backtest.commission_bps))
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.net_returns.to_frame("net").join(result.gross_returns.rename("gross")).to_parquet(
        out_dir / "backtest_returns.parquet"
    )
    positions.to_frame().to_parquet(out_dir / "positions.parquet")
    summary = {**result.metrics, **check}
    pd.Series(summary).to_json(out_dir / "backtest_metrics.json")
    print(summary)


if __name__ == "__main__":
    main()
