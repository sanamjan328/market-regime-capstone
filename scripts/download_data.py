#!/usr/bin/env python
"""Download primary SPY panel (+ optional BTC) and write parquet."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    from src.data import build_btc_panel, build_primary_panel

    print(OmegaConf.to_yaml(cfg.data))
    panel = build_primary_panel(
        universe=cfg.data.universe,
        start=cfg.data.start,
        end=cfg.data.end,
        cross_asset=list(cfg.data.cross_asset),
        fred_series=OmegaConf.to_container(cfg.data.fred_series, resolve=True),
        raw_dir=cfg.data.raw_dir,
        processed_dir=cfg.data.processed_dir,
        macro_lag_days_default=cfg.data.macro_lag_days_default,
    )
    print(f"Primary panel: {panel.shape} -> {cfg.data.processed_dir}")

    if cfg.data.secondary.get("enabled", True):
        btc = build_btc_panel(
            start=cfg.data.secondary.start,
            end=cfg.data.secondary.end,
            interval=cfg.data.secondary.interval,
            raw_dir=cfg.data.raw_dir,
            processed_dir=cfg.data.processed_dir,
        )
        print(f"BTC panel: {btc.shape}")


if __name__ == "__main__":
    main()
