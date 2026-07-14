#!/usr/bin/env python
"""
Day-19: Final holdout — 2022-2024, touched EXACTLY ONCE.

Trains each model on all pre-2022 data and predicts on 2022-2024.
Run this script only when all modelling decisions are final.
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig


MODELS_TO_EVAL = [
    "buy_hold",
    "zero",
    "arima_garch",
    "lightgbm",
    "dlinear",
    "patchtst_no_regime",
    "patchtst_film",
]


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    from src.backtest.costs import total_cost
    from src.backtest.metrics import calmar, max_drawdown, profit_factor, sharpe, sortino
    from src.backtest.sizing import kelly_lite_positions
    from src.eval.bootstrap import bootstrap_sharpe_ci
    from src.eval.dsr import deflated_sharpe_ratio
    from src.eval.regime_attribution import per_regime_attribution
    from src.features.cv import holdout_indices
    from src.features.scalers import ExpandingStandardScaler, winsorize
    from src.models.baselines import buy_and_hold, fit_arima_garch, fit_lightgbm, zero_forecast
    from src.models.dataset import SequenceDataset
    from src.models.dlinear import DLinear
    from src.models.patchtst import PatchTST
    from src.models.train_loop import predict_loader, train_one

    feat_path = Path(cfg.data.processed_dir) / f"{cfg.data.universe.lower()}_features_regimes.parquet"
    panel_path = Path(cfg.data.processed_dir) / f"{cfg.data.universe.lower()}_panel.parquet"
    out_dir = Path(cfg.output_dir)

    feats = pd.read_parquet(feat_path)
    panel = pd.read_parquet(panel_path)
    dates = pd.DatetimeIndex(feats.index)

    h_idx = holdout_indices(dates, start=cfg.cv.final_holdout_start, end=cfg.cv.final_holdout_end)
    holdout_start_ts = pd.Timestamp(cfg.cv.final_holdout_start)
    train_idx = np.where(dates < holdout_start_ts)[0]

    print(f"Train: {dates[train_idx[0]].date()} → {dates[train_idx[-1]].date()}  ({len(train_idx)} bars)")
    print(f"Holdout: {dates[h_idx[0]].date()} → {dates[h_idx[-1]].date()}  ({len(h_idx)} bars)")
    print("=" * 60)

    regime_cols = [c for c in feats.columns if c.startswith("p_regime_")]
    drop_cols = {"y_vol_norm", "y_triple_barrier", "fwd_ret_1", "open_next", *regime_cols}
    feature_cols = [c for c in feats.columns if c not in drop_cols]

    target = feats["y_vol_norm"].values
    fwd = feats["fwd_ret_1"].values if "fwd_ret_1" in feats.columns else target
    regimes = feats[regime_cols].values if regime_cols else None
    n_regimes = len(regime_cols) if regime_cols else int(cfg.model.get("n_regimes", 3))

    scaler = ExpandingStandardScaler()
    X_train_df = winsorize(feats.iloc[train_idx][feature_cols])
    scaler.fit(X_train_df)
    X_all = scaler.transform(winsorize(feats[feature_cols], ref=X_train_df)).values.astype(np.float32)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    lookback = int(cfg.model.get("lookback", 60))
    n_features = X_all.shape[1]
    rv_all = feats["realized_vol_20d"].ffill().fillna(0.01).values

    def _net_returns(mu, sigma, idx_set):
        rv_h = rv_all[idx_set]
        reg_h = regimes[idx_set] if regimes is not None else None
        pos = kelly_lite_positions(
            mu, sigma, rv_h, reg_h,
            vol_target=float(cfg.backtest.vol_target),
            leverage_cap=float(cfg.backtest.leverage_cap),
        )
        executed = pd.Series(pos, index=dates[idx_set]).shift(1).fillna(0.0)
        bar_ret = panel["close"].reindex(dates[idx_set]).pct_change().fillna(0.0)
        gross = executed.values * bar_ret.values
        costs = total_cost(executed.values, commission_bps=float(cfg.backtest.commission_bps))
        return pd.Series(gross - costs, index=dates[idx_set])

    results: dict = {}
    n_trials = len(MODELS_TO_EVAL) * int(cfg.get("n_seeds", 5))

    for model_name in MODELS_TO_EVAL:
        print(f"\n  [{model_name}]")

        # ── Baselines ────────────────────────────────────────────────────────
        if model_name == "buy_hold":
            mu = np.ones(len(h_idx))
            sigma = rv_all[h_idx]

        elif model_name == "zero":
            mu = np.zeros(len(h_idx))
            sigma = rv_all[h_idx]

        elif model_name == "arima_garch":
            raw_ret = pd.Series(fwd, index=dates)
            bp = fit_arima_garch(
                raw_ret.iloc[train_idx], raw_ret.iloc[h_idx],
                realized_vol_test=rv_all[h_idx], refit_every=40,
            )
            mu, sigma = bp.mu, bp.sigma

        elif model_name == "lightgbm":
            bp = fit_lightgbm(X_all[train_idx], target[train_idx], X_all[h_idx], rv_all[h_idx])
            mu, sigma = bp.mu, bp.sigma

        # ── DLinear ──────────────────────────────────────────────────────────
        elif model_name == "dlinear":
            model = DLinear(lookback=lookback, n_features=n_features).to(device)
            train_ds = SequenceDataset(X_all, target, regimes, lookback, train_idx)
            res = train_one(model, train_ds, device, use_s=False,
                            lr=float(cfg.model.get("lr", 1e-3)),
                            max_epochs=int(cfg.model.get("max_epochs", 50)),
                            patience=int(cfg.model.get("patience", 8)))
            model.load_state_dict(res.best_state)
            test_ds = SequenceDataset(X_all, target, regimes, lookback, h_idx)
            loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)
            mu_arr, sig_arr, _ = predict_loader(model, loader, device, use_s=False)
            mu, sigma = mu_arr, sig_arr
            h_idx = test_ds.indices  # valid subset

        # ── PatchTST (no-regime) ──────────────────────────────────────────────
        elif model_name == "patchtst_no_regime":
            model = PatchTST(
                n_features=n_features, lookback=lookback,
                n_regimes=n_regimes, use_regime=False, conditioning="none",
                d_model=int(cfg.model.get("d_model", 128)),
                n_heads=int(cfg.model.get("n_heads", 4)),
                n_layers=int(cfg.model.get("n_layers", 3)),
                dropout=float(cfg.model.get("dropout", 0.2)),
            ).to(device)
            train_ds = SequenceDataset(X_all, target, regimes, lookback, train_idx)
            res = train_one(model, train_ds, device, use_s=False,
                            lr=float(cfg.model.get("lr", 1e-3)),
                            max_epochs=int(cfg.model.get("max_epochs", 50)),
                            patience=int(cfg.model.get("patience", 8)))
            model.load_state_dict(res.best_state)
            test_ds = SequenceDataset(X_all, target, regimes, lookback, h_idx)
            loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)
            mu_arr, sig_arr, _ = predict_loader(model, loader, device, use_s=False)
            mu, sigma = mu_arr, sig_arr
            h_idx = test_ds.indices

        # ── PatchTST + FiLM ──────────────────────────────────────────────────
        elif model_name == "patchtst_film":
            mu_seeds, sig_seeds = [], []
            n_seeds = int(cfg.get("n_seeds", 5))
            for seed_i in range(n_seeds):
                seed = int(cfg.get("seed", 42)) + seed_i
                np.random.seed(seed)
                torch.manual_seed(seed)
                model = PatchTST(
                    n_features=n_features, lookback=lookback,
                    n_regimes=n_regimes, use_regime=True, conditioning="film",
                    d_model=int(cfg.model.get("d_model", 128)),
                    n_heads=int(cfg.model.get("n_heads", 4)),
                    n_layers=int(cfg.model.get("n_layers", 3)),
                    dropout=float(cfg.model.get("dropout", 0.2)),
                    film_hidden=int(cfg.model.get("film_hidden", 64)),
                ).to(device)
                train_ds = SequenceDataset(X_all, target, regimes, lookback, train_idx)
                res = train_one(model, train_ds, device, use_s=True,
                                lr=float(cfg.model.get("lr", 1e-3)),
                                max_epochs=int(cfg.model.get("max_epochs", 50)),
                                patience=int(cfg.model.get("patience", 8)))
                model.load_state_dict(res.best_state)
                test_ds = SequenceDataset(X_all, target, regimes, lookback, h_idx)
                loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)
                m_arr, s_arr, _ = predict_loader(model, loader, device, use_s=True)
                mu_seeds.append(m_arr)
                sig_seeds.append(s_arr)
            mu = np.stack(mu_seeds).mean(axis=0)
            sigma = np.stack(sig_seeds).mean(axis=0)
            h_idx = test_ds.indices
            print(f"    5-seed ensemble complete")

        else:
            continue

        net_rets = _net_returns(mu, sigma, h_idx)
        mdd, dd_dur = max_drawdown(net_rets)
        ci = bootstrap_sharpe_ci(net_rets, n_boot=500, mean_block=10, seed=int(cfg.seed))
        skew = float(net_rets.skew())
        kurt = float(net_rets.kurtosis() + 3)
        dsr = deflated_sharpe_ratio(ci["sharpe"], n_obs=len(net_rets), n_trials=n_trials, skew=skew, kurt=kurt)

        attr = None
        if regimes is not None:
            regime_df = feats.loc[dates[h_idx], regime_cols]
            attr = per_regime_attribution(net_rets, regime_df)

        results[model_name] = {
            "n_obs": len(net_rets),
            "date_range": [str(dates[h_idx[0]].date()), str(dates[h_idx[-1]].date())],
            "sharpe": ci["sharpe"],
            "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"],
            "deflated_sharpe": dsr,
            "sortino": sortino(net_rets),
            "calmar": calmar(net_rets),
            "max_dd": mdd,
            "dd_duration": int(dd_dur),
            "profit_factor": profit_factor(net_rets),
            "ann_return": float(net_rets.mean() * 252),
            "ann_vol": float(net_rets.std() * np.sqrt(252)),
            "regime_attribution": attr.to_dict("records") if attr is not None else None,
        }
        ci_str = f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"
        print(f"    Sharpe={ci['sharpe']:+.3f} {ci_str}  DSR={dsr:.3f}  MaxDD={mdd:.3f}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    holdout_json = out_dir / "final_holdout.json"
    with open(holdout_json, "w") as f:
        json.dump(results, f, indent=2, default=str)

    rows = [
        {
            "model": name,
            "sharpe": m["sharpe"],
            "ci_low": m["ci_low"],
            "ci_high": m["ci_high"],
            "dsr": m["deflated_sharpe"],
            "sortino": m["sortino"],
            "calmar": m["calmar"],
            "max_dd": m["max_dd"],
            "ann_return": m["ann_return"],
            "n_obs": m["n_obs"],
        }
        for name, m in results.items()
    ]
    summary_df = pd.DataFrame(rows)
    summary_csv = out_dir / "final_holdout_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n── Final holdout summary ──")
    print(summary_df[["model", "sharpe", "ci_low", "ci_high", "dsr", "max_dd", "ann_return"]].to_string(index=False))
    print(f"\nWrote {holdout_json}")
    print(f"Wrote {summary_csv}")

    film = results.get("patchtst_film", {})
    ctrl = results.get("patchtst_no_regime", {})
    if film and ctrl:
        delta = film.get("sharpe", 0) - ctrl.get("sharpe", 0)
        print(f"\nThesis delta (FiLM − no_regime): {delta:+.3f}")
        if delta > 0:
            print("  FiLM beats control on holdout.")
        else:
            print("  FiLM does NOT beat control on holdout. Diagnose: regime posteriors, "
                  "FiLM divergence, cost drag.")


if __name__ == "__main__":
    main()
