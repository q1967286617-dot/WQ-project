"""Train a CatBoost binary classifier.

Usage:
    python scripts/run_train_catboost.py --run_id catboost_v1
    python scripts/run_train_catboost.py --run_id catboost_v1 --model_cfg configs/model.yaml

Model artifact is saved to models/catboost_<run_id>.joblib.
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1]))

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.utils.paths import load_yaml, resolve_paths, ensure_dir
from src.utils.logging import get_logger
from src.modeling.preprocess import (
    fit_train_imputer_and_scaler,
    apply_imputer_and_scaler,
    split_xy,
)
from src.modeling.train_catboost import (
    CatBoostArtifacts,
    train_catboost_binary,
    save_artifacts,
    _prepare_cat_strings,
)

logger = get_logger("run_train_catboost")


def _read_df(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def main() -> None:
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

    train_df = _read_df(paths.processed_dir / "train.parquet")
    val_df = _read_df(paths.processed_dir / "val.parquet")

    target_col = model_cfg.get("target_col", "y_div_10d")
    num_cols = list(model_cfg["num_cols"])
    cat_cols = list(model_cfg["cat_cols"])
    hp = model_cfg.get("hyperparams", {})

    # CatBoost is tree-based: impute but do NOT scale
    stats = fit_train_imputer_and_scaler(train_df, num_cols)
    train_df = apply_imputer_and_scaler(train_df, num_cols, stats, do_scale=False)
    val_df = apply_imputer_and_scaler(val_df, num_cols, stats, do_scale=False)

    # CatBoost takes raw strings for categoricals
    train_df = _prepare_cat_strings(train_df, cat_cols)
    val_df = _prepare_cat_strings(val_df, cat_cols)

    feature_names = num_cols + cat_cols
    X_train, y_train = split_xy(train_df, num_cols, cat_cols, target_col)
    X_val, y_val = split_xy(val_df, num_cols, cat_cols, target_col)

    # Convert cat cols to object so CatBoost sees them as strings, not categories
    for c in cat_cols:
        X_train[c] = X_train[c].astype(str)
        X_val[c] = X_val[c].astype(str)

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    logger.info(f"train pos={pos}, neg={neg}, ratio={pos/(pos+neg):.3f}")

    model = train_catboost_binary(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        cat_cols=cat_cols,
        pos=pos,
        neg=neg,
        learning_rate=float(hp.get("learning_rate", 0.05)),
        iterations=int(hp.get("num_boost_round", 500)),
        early_stopping_rounds=int(hp.get("early_stopping_rounds", 20)),
    )

    art = CatBoostArtifacts(
        model=model,
        impute_stats=stats,
        num_cols=num_cols,
        cat_cols=cat_cols,
        target_col=target_col,
        feature_names=feature_names,
        do_scale=False,
    )

    art_path = paths.models_dir / f"catboost_{run_id}.joblib"
    save_artifacts(art, art_path)
    logger.info(f"saved artifacts: {art_path}")
    (run_dir / "meta.txt").write_text(f"model_artifacts={art_path}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
