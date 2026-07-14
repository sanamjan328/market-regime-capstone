#!/usr/bin/env python
"""Build feature matrix from processed panel parquet."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    from src.data import load_panel
    from src.features import build_feature_matrix

    panel_path = Path(cfg.data.processed_dir) / f"{cfg.data.universe.lower()}_panel.parquet"
    panel = load_panel(panel_path)
    feats = build_feature_matrix(panel)
    out = Path(cfg.data.processed_dir) / f"{cfg.data.universe.lower()}_features.parquet"
    feats.to_parquet(out)
    print(f"Features: {feats.shape} d_fracdiff={feats.attrs.get('fracdiff_d')} -> {out}")


if __name__ == "__main__":
    main()
