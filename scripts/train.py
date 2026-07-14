#!/usr/bin/env python
"""Train PatchTST / DLinear / hard-switch with val-Sharpe early stop + seed ensemble."""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(cfg: DictConfig, n_features: int, n_regimes: int, device: torch.device):
    from src.models.dlinear import DLinear
    from src.models.hard_switch import HardSwitchPatchTST
    from src.models.patchtst import PatchTST

    name = str(cfg.model.get("name", "patchtst_film"))
    lookback = int(cfg.model.lookback)
    loss_type = str(cfg.model.get("loss_type", "gaussian"))
    conditioning = str(cfg.model.get("conditioning", "film"))

    if name == "dlinear":
        return DLinear(lookback=lookback, n_features=n_features).to(device), False

    common = dict(
        n_features=n_features,
        lookback=lookback,
        patch_len=int(cfg.model.get("patch_len", 16)),
        stride=int(cfg.model.get("stride", 8)),
        d_model=int(cfg.model.get("d_model", 128)),
        n_heads=int(cfg.model.get("n_heads", 4)),
        n_layers=int(cfg.model.get("n_layers", 3)),
        dropout=float(cfg.model.get("dropout", 0.2)),
        ff_dim=int(cfg.model.get("ff_dim", 256)),
        n_regimes=n_regimes,
        film_hidden=int(cfg.model.get("film_hidden", 64)),
        loss_type=loss_type,
    )

    if name == "patchtst_hard_switch" or conditioning == "hard":
        model = HardSwitchPatchTST(**common).to(device)
        return model, True

    model = PatchTST(
        use_regime=bool(cfg.model.get("use_regime", True)),
        conditioning=conditioning,
        **common,
    ).to(device)
    use_s = bool(cfg.model.get("use_regime", True)) and conditioning != "none"
    return model, use_s


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    from src.features.cv import purged_walk_forward
    from src.features.scalers import ExpandingStandardScaler, winsorize
    from src.models.dataset import SequenceDataset
    from src.models.train_loop import predict_loader, train_one

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
    regimes = df[regime_cols].values if regime_cols else None
    fwd = df["fwd_ret_1"].values if "fwd_ret_1" in df.columns else target

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
    lookback = int(cfg.model.lookback)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device={device}")
    out_dir = Path(cfg.output_dir) / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    name = str(cfg.model.get("name", "patchtst_film"))
    loss_type = str(cfg.model.get("loss_type", "gaussian"))
    n_seeds = int(cfg.get("n_seeds", 1))
    base_seed = int(cfg.get("seed", 42))

    seed_frames: list[pd.DataFrame] = []
    for seed_i in range(n_seeds):
        seed = base_seed + seed_i
        _set_seed(seed)
        print(f"=== seed {seed} ({seed_i + 1}/{n_seeds}) model={name} ===")
        all_preds = []
        for fold in folds:
            scaler = ExpandingStandardScaler()
            X_train_df = winsorize(df.iloc[fold.train_idx][feature_cols])
            scaler.fit(X_train_df)
            X_all = scaler.transform(winsorize(df[feature_cols], ref=X_train_df)).values.astype(
                np.float32
            )

            train_ds = SequenceDataset(X_all, target, regimes, lookback, fold.train_idx)
            test_ds = SequenceDataset(X_all, target, regimes, lookback, fold.test_idx)
            if len(train_ds) == 0 or len(test_ds) == 0:
                print(f"skip fold {fold.fold_id}: empty dataset")
                continue

            n_features = X_all.shape[1]
            n_regimes = regimes.shape[1] if regimes is not None else int(cfg.model.get("n_regimes", 3))
            model, use_s = _build_model(cfg, n_features, n_regimes, device)

            result = train_one(
                model,
                train_ds,
                device,
                use_s=use_s,
                lr=float(cfg.model.get("lr", 1e-3)),
                weight_decay=float(cfg.model.get("weight_decay", 1e-4)),
                batch_size=int(cfg.model.get("batch_size", 64)),
                max_epochs=int(cfg.model.get("max_epochs", 50)),
                patience=int(cfg.model.get("patience", 8)),
                val_frac=float(cfg.model.get("val_frac", 0.2)),
                loss_type=loss_type,
            )
            model.load_state_dict(result.best_state)
            print(
                f"fold {fold.fold_id} seed={seed} epochs={result.epochs_run} "
                f"val_sharpe={result.best_metric:.4f}"
            )

            test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
            mu_arr, sig_arr, _ = predict_loader(
                model, test_loader, device, use_s=use_s, loss_type=loss_type
            )
            pred_idx = test_ds.indices
            part = pd.DataFrame(
                {
                    "mu": mu_arr,
                    "sigma": sig_arr,
                    "y": target[pred_idx],
                    "fwd_ret": fwd[pred_idx],
                    "fold": fold.fold_id,
                    "seed": seed,
                },
                index=dates[pred_idx],
            )
            all_preds.append(part)
            ckpt = out_dir / f"{name}_seed{seed}_fold{fold.fold_id}.pt"
            torch.save(model.state_dict(), ckpt)

        if all_preds:
            seed_frames.append(pd.concat(all_preds).sort_index())

    if not seed_frames:
        print("No predictions produced")
        return

    # Average mu/sigma across seeds on intersecting dates
    aligned = seed_frames[0][["mu", "sigma", "y", "fwd_ret", "fold"]].copy()
    if len(seed_frames) > 1:
        mu_stack = np.stack([f.reindex(aligned.index)["mu"].values for f in seed_frames], axis=0)
        sig_stack = np.stack([f.reindex(aligned.index)["sigma"].values for f in seed_frames], axis=0)
        aligned["mu"] = np.nanmean(mu_stack, axis=0)
        aligned["sigma"] = np.nanmean(sig_stack, axis=0)

    pred_path = out_dir / f"{name}_oos.parquet"
    aligned.to_parquet(pred_path)
    print(f"Wrote ensembled predictions {aligned.shape} ({n_seeds} seeds) -> {pred_path}")


if __name__ == "__main__":
    main()
