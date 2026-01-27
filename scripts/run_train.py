from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1]))

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from src.utils.paths import load_yaml, resolve_paths, ensure_dir
from src.utils.logging import get_logger
from src.modeling.preprocess import fit_train_imputer_and_scaler, apply_imputer_and_scaler, prepare_categorical, split_xy
from src.modeling.train import build_dmatrix, train_xgb_binary, TrainArtifacts, save_artifacts


logger = get_logger("run_train")

def _read_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to read parquet: {path}. Install pyarrow (recommended) or convert to CSV.\n"
                f"Original error: {e}"
            )
    return pd.read_csv(path)


def _read_split_parquets(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_p = processed_dir / "train.parquet"
    val_p = processed_dir / "val.parquet"
    test_p = processed_dir / "test.parquet"
    if not (train_p.exists() and val_p.exists() and test_p.exists()):
        raise FileNotFoundError(
            "Missing processed split files. Expected:\n"
            f"  {train_p}\n  {val_p}\n  {test_p}\n\n"
            "Place your prepared splits there or extend scripts/run_train.py to build from raw."
        )
    return _read_df(train_p), _read_df(val_p), _read_df(test_p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--model_cfg", default="configs/model.yaml")
    ap.add_argument("--run_id", default=None)
    args = ap.parse_args()

    run_id = args.run_id or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    paths_cfg = load_yaml(Path(args.paths))
    model_cfg = load_yaml(Path(args.model_cfg))

    paths = resolve_paths(paths_cfg, project_root=Path.cwd())
    run_dir = paths.outputs_dir / run_id
    ensure_dir(run_dir)
    ensure_dir(paths.models_dir)

    logger.info(f"run_id={run_id}")
    logger.info(f"processed_dir={paths.processed_dir}")

    # Load processed splits
    train_df, val_df, test_df = _read_split_parquets(paths.processed_dir)

    target_col = model_cfg.get("target_col", "y_div_10d")
    num_cols = list(model_cfg["num_cols"])
    cat_cols = list(model_cfg["cat_cols"])

    hyperparams = model_cfg['hyperparams']

    # Fit imputer/scaler on TRAIN only, then apply to all
    stats = fit_train_imputer_and_scaler(train_df, num_cols)
    train_df = apply_imputer_and_scaler(train_df, num_cols, stats, do_scale=model_cfg['do_scale'])
    val_df = apply_imputer_and_scaler(val_df, num_cols, stats, do_scale=model_cfg['do_scale'])
    test_df = apply_imputer_and_scaler(test_df, num_cols, stats, do_scale=model_cfg['do_scale'])

    # Prepare categorical with TRAIN categories
    train_df, val_df, test_df = prepare_categorical(train_df, val_df, test_df, cat_cols)
    cat_categories = {c: train_df[c].cat.categories.astype(str).tolist() for c in cat_cols}

    # Split to X/y
    X_train, y_train = split_xy(train_df, num_cols, cat_cols, target_col)
    X_val, y_val = split_xy(val_df, num_cols, cat_cols, target_col)

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)

    feature_names = num_cols + cat_cols

    dtrain = build_dmatrix(X_train, y_train, enable_categorical=True, feature_names=feature_names)
    dval = build_dmatrix(X_val, y_val, enable_categorical=True, feature_names=feature_names)

    booster = train_xgb_binary(
        dtrain=dtrain,
        dval=dval,
        pos=pos,
        neg=neg,
        max_depth=hyperparams['max_depth'],
        learning_rate=hyperparams['learning_rate'],
        num_boost_round=hyperparams['num_boost_round'],
        early_stopping_rounds=hyperparams['early_stopping_rounds'],
        verbose_eval=50,
    )

    art = TrainArtifacts(
        booster=booster,
        impute_stats=stats,
        num_cols=num_cols,
        cat_cols=cat_cols,
        cat_categories=cat_categories,
        target_col=target_col,
        feature_names=feature_names,
        do_scale=model_cfg['do_scale'],
    )

    art_path = paths.models_dir / f"xgb_{run_id}.joblib"
    save_artifacts(art, art_path)
    logger.info(f"saved artifacts: {art_path}")

    (run_dir / "meta.txt").write_text(f"model_artifacts={art_path}\n", encoding="utf-8")
    logger.info(f"run_dir: {run_dir}")


if __name__ == "__main__":
    main()
