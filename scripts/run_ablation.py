#!/usr/bin/env python
"""Build the full Week-2 ablation table (models 0-7) from OOS prediction files."""

from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

# Map saved prediction stems -> ablation model names
PRED_TO_ABLATION = {
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

    pred_dir = Path(cfg.output_dir) / "predictions"
    predictions: dict[str, pd.DataFrame] = {}
    missing = []

    for stem, abl_name in PRED_TO_ABLATION.items():
        path = pred_dir / f"{stem}_oos.parquet"
        if not path.exists():
            if abl_name in {m for m, _ in ABLATION_SPEC}:
                missing.append(abl_name)
            continue
        df = pd.read_parquet(path)
        if "fwd_ret" not in df.columns:
            # fallback: attach from features if available
            feat = Path(cfg.data.processed_dir) / f"{cfg.data.universe.lower()}_features_regimes.parquet"
            if feat.exists():
                raw = pd.read_parquet(feat)
                if "fwd_ret_1" in raw.columns:
                    df = df.copy()
                    df["fwd_ret"] = raw["fwd_ret_1"].reindex(df.index).values
        predictions[abl_name] = df
        print(f"loaded {stem} -> {abl_name}: {df.shape}")

    if not predictions:
        print(f"No prediction files found in {pred_dir}")
        return

    table = build_ablation_table(predictions)
    # Stable order matching the architecture ablation list
    order = [m for m, _ in ABLATION_SPEC]
    table["model"] = pd.Categorical(table["model"], categories=order, ordered=True)
    table = table.sort_values("model").reset_index(drop=True)

    out = Path(cfg.output_dir) / "ablation_table.csv"
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    if missing:
        print(f"Missing models (not yet trained): {sorted(set(missing) - set(predictions))}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
