#!/usr/bin/env python
"""Run Day-8 classical/ML baselines and write OOS prediction parquets."""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    from src.features.cv import purged_walk_forward
    from src.features.scalers import ExpandingStandardScaler, winsorize
    from src.models.baselines import buy_and_hold, fit_arima_garch, fit_lightgbm, zero_forecast

    path = Path(cfg.data.processed_dir) / f"{cfg.data.universe.lower()}_features_regimes.parquet"
    if not path.exists():
        path = Path(cfg.data.processed_dir) / f"{cfg.data.universe.lower()}_features.parquet"
    df = pd.read_parquet(path)

    regime_cols = [c for c in df.columns if c.startswith("p_regime_")]
    drop = {
        "y_vol_norm",
        "y_triple_barrier",
        "fwd_ret_1",
        "open_next",
        *regime_cols,
    }
    feature_cols = [c for c in df.columns if c not in drop]
    target = df["y_vol_norm"].values
    fwd = df["fwd_ret_1"].values if "fwd_ret_1" in df.columns else target
    # raw returns for ARIMA-GARCH (prefer close-to-close if present)
    if "r_1" in df.columns:
        raw_ret = df["r_1"]
    else:
        raw_ret = pd.Series(fwd, index=df.index)

    dates = pd.DatetimeIndex(df.index)
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
    out_dir = Path(cfg.output_dir) / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, list[pd.DataFrame]] = {
        "buy_hold": [],
        "zero": [],
        "arima_garch": [],
        "lightgbm": [],
    }

    for fold in folds:
        tr, te = fold.train_idx, fold.test_idx
        rv_te = df["realized_vol_20d"].iloc[te].ffill().fillna(0.01).values

        bh = buy_and_hold(len(te), rv_te)
        zf = zero_forecast(len(te), rv_te)
        buckets["buy_hold"].append(
            pd.DataFrame(
                {"mu": bh.mu, "sigma": bh.sigma, "y": target[te], "fwd_ret": fwd[te], "fold": fold.fold_id},
                index=dates[te],
            )
        )
        buckets["zero"].append(
            pd.DataFrame(
                {"mu": zf.mu, "sigma": zf.sigma, "y": target[te], "fwd_ret": fwd[te], "fold": fold.fold_id},
                index=dates[te],
            )
        )

        # ARIMA-GARCH on raw returns
        ag = fit_arima_garch(
            raw_ret.iloc[tr],
            raw_ret.iloc[te],
            realized_vol_test=rv_te,
            refit_every=40,
        )
        buckets["arima_garch"].append(
            pd.DataFrame(
                {"mu": ag.mu, "sigma": ag.sigma, "y": target[te], "fwd_ret": fwd[te], "fold": fold.fold_id},
                index=dates[te],
            )
        )

        # LightGBM on scaled tabular features (identical feature set)
        scaler = ExpandingStandardScaler()
        X_train_df = winsorize(df.iloc[tr][feature_cols])
        scaler.fit(X_train_df)
        X_all = scaler.transform(winsorize(df[feature_cols], ref=X_train_df)).values.astype(np.float32)
        lgbm = fit_lightgbm(X_all[tr], target[tr], X_all[te], rv_te)
        buckets["lightgbm"].append(
            pd.DataFrame(
                {
                    "mu": lgbm.mu,
                    "sigma": lgbm.sigma,
                    "y": target[te],
                    "fwd_ret": fwd[te],
                    "fold": fold.fold_id,
                },
                index=dates[te],
            )
        )
        print(f"fold {fold.fold_id}: baselines done (n_test={len(te)})")

    for name, parts in buckets.items():
        pred = pd.concat(parts).sort_index()
        out = out_dir / f"{name}_oos.parquet"
        pred.to_parquet(out)
        print(f"Wrote {name} {pred.shape} -> {out}")


if __name__ == "__main__":
    main()
