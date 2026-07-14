#!/usr/bin/env python
"""Fit HMM regimes with causal filtered posteriors across walk-forward folds."""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    from src.features.cv import purged_walk_forward
    from src.regimes.hmm import fit_filter_fold, regime_feature_frame, select_k

    feat_path = Path(cfg.data.processed_dir) / f"{cfg.data.universe.lower()}_features.parquet"
    feats = pd.read_parquet(feat_path)
    regime_cols = list(cfg.cv.regime_features)
    available = [c for c in regime_cols if c in feats.columns]
    X = feats[available].replace([np.inf, -np.inf], np.nan)
    # fill warm-up with column medians for HMM fitting only
    X = X.fillna(X.median())

    dates = pd.DatetimeIndex(feats.index)
    folds = list(
        purged_walk_forward(
            dates,
            initial_train_end=cfg.cv.initial_train_end,
            test_years=cfg.cv.test_years,
            embargo_days=cfg.cv.embargo_days,
            purge_horizon_days=cfg.cv.purge_horizon_days,
            final_holdout_start=cfg.cv.final_holdout_start,
        )
    )
    print(f"Folds: {len(folds)}  regime features: {available}")

    # select K on first fold train
    k, scores = select_k(
        X.values[folds[0].train_idx],
        k_grid=list(cfg.cv.n_regime_states_grid),
        vol_index=0,
    )
    if cfg.cv.selected_k:
        k = int(cfg.cv.selected_k)
    print(f"Selected K={k}  scores={scores}")

    # store filtered posteriors; later folds overwrite overlapping train region with latest fit
    post = np.full((len(feats), k), np.nan)
    for fold in folds:
        result = fit_filter_fold(
            X.values,
            fold.train_idx,
            fold.test_idx,
            n_components=k,
            random_state=cfg.seed,
            vol_index=0,
        )
        # write train+test filtered values from this fold's frozen model
        end = int(fold.test_idx[-1]) + 1
        post[:end] = result.filtered
        # for OOS integrity, prefer test-segment write as authoritative for that period
        post[fold.test_idx] = result.filtered[fold.test_idx]
        print(
            f"fold {fold.fold_id}: train_end={fold.train_end.date()} "
            f"test={fold.test_start.date()}→{fold.test_end.date()} "
            f"bic={result.bic:.1f} dur={result.mean_duration:.1f}"
        )

    regime_df = regime_feature_frame(dates, np.nan_to_num(post, nan=1.0 / k), prefix="p_regime")
    # pad columns if needed
    out = feats.join(regime_df)
    out_path = Path(cfg.data.processed_dir) / f"{cfg.data.universe.lower()}_features_regimes.parquet"
    out.to_parquet(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
