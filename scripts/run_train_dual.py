"""
Dual-booster training: a binary classifier on y_div_10d plus a regression
booster on fwd_ret_Hd. Both boosters share the same preprocessing
(imputer/scaler + train-fitted categorical schema) and the same feature
matrix, so downstream fusion only needs to combine two scores per row.

Artifacts are saved separately:
    models/xgb_<run_id>.joblib        <- binary classifier (TrainArtifacts)
    models/xgb_reg_<run_id>.joblib    <- regressor        (RegArtifacts)

Both filenames match the conventions used by run_predict.py / scripts that
already look for `xgb_<run_id>.joblib`, so the classifier half is a drop-in
replacement for run_train.py's output.
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1]))

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import load_yaml, resolve_paths, ensure_dir
from src.utils.logging import get_logger
from src.modeling.preprocess import (
    fit_train_imputer_and_scaler,
    apply_imputer_and_scaler,
    prepare_categorical,
    split_xy,
)
from src.modeling.train import (
    build_dmatrix,
    train_xgb_binary,
    train_xgb_regressor,
    train_xgb_ranker,
    TrainArtifacts,
    RegArtifacts,
    save_artifacts,
)


logger = get_logger("run_train_dual")


def _read_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _read_split_parquets(processed_dir: Path):
    train_p = processed_dir / "train.parquet"
    val_p = processed_dir / "val.parquet"
    test_p = processed_dir / "test.parquet"
    if not (train_p.exists() and val_p.exists() and test_p.exists()):
        raise FileNotFoundError(
            f"Missing processed splits under {processed_dir}. "
            "Run scripts/run_build_dataset.py first."
        )
    return _read_df(train_p), _read_df(val_p), _read_df(test_p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--model_cfg", default="configs/model.yaml")
    ap.add_argument("--run_id", default=None)
    ap.add_argument(
        "--reg_target",
        default=None,
        help="Regression target column (default: fwd_ret_{H_label}d). "
             "Override if you built a different horizon.",
    )
    ap.add_argument("--reg_num_boost_round", type=int, default=None)
    ap.add_argument("--reg_max_depth", type=int, default=None)
    ap.add_argument("--reg_learning_rate", type=float, default=None)
    ap.add_argument("--cls_num_boost_round", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42, help="Seed for both cls and reg boosters.")
    ap.add_argument(
        "--reg_mode",
        choices=["regression", "rank_ndcg", "rank_pairwise"],
        default="regression",
        help="'regression' = MSE on fwd_ret (original); "
             "'rank_ndcg'/'rank_pairwise' = learning-to-rank, groups=trading day, "
             "labels=per-day quantile bin of fwd_ret",
    )
    ap.add_argument("--rank_n_bins", type=int, default=5,
                    help="Number of per-day quantile bins (relevance levels) for rank modes.")
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

    train_df, val_df, test_df = _read_split_parquets(paths.processed_dir)

    target_col = model_cfg.get("target_col", "y_div_10d")
    # Default reg target: if H_label is not exposed via model_cfg, guess from binary target name.
    reg_target_default = (
        args.reg_target
        or model_cfg.get("reg_target_col")
        or target_col.replace("y_div_", "fwd_ret_")
    )
    reg_target = reg_target_default

    num_cols = list(model_cfg["num_cols"])
    cat_cols = list(model_cfg["cat_cols"])

    hp = dict(model_cfg["hyperparams"])
    if args.cls_num_boost_round is not None:
        hp["num_boost_round"] = args.cls_num_boost_round
    reg_hp = dict(hp)
    if args.reg_num_boost_round is not None:
        reg_hp["num_boost_round"] = args.reg_num_boost_round
    if args.reg_max_depth is not None:
        reg_hp["max_depth"] = args.reg_max_depth
    if args.reg_learning_rate is not None:
        reg_hp["learning_rate"] = args.reg_learning_rate

    # ── shared preprocessing ────────────────────────────────────────────
    stats = fit_train_imputer_and_scaler(train_df, num_cols)
    train_df = apply_imputer_and_scaler(train_df, num_cols, stats, do_scale=model_cfg["do_scale"])
    val_df = apply_imputer_and_scaler(val_df, num_cols, stats, do_scale=model_cfg["do_scale"])
    test_df = apply_imputer_and_scaler(test_df, num_cols, stats, do_scale=model_cfg["do_scale"])

    train_df, val_df, test_df = prepare_categorical(train_df, val_df, test_df, cat_cols)
    cat_categories = {c: train_df[c].cat.categories.astype(str).tolist() for c in cat_cols}
    feature_names = num_cols + cat_cols

    # ── classifier branch ──────────────────────────────────────────────
    X_tr_cls, y_tr_cls = split_xy(train_df, num_cols, cat_cols, target_col)
    X_va_cls, y_va_cls = split_xy(val_df, num_cols, cat_cols, target_col)

    pos = int(y_tr_cls.sum())
    neg = int(len(y_tr_cls) - pos)

    d_tr_cls = build_dmatrix(X_tr_cls, y_tr_cls, enable_categorical=True, feature_names=feature_names)
    d_va_cls = build_dmatrix(X_va_cls, y_va_cls, enable_categorical=True, feature_names=feature_names)

    logger.info("training classifier: target=%s, pos=%d, neg=%d", target_col, pos, neg)
    booster_cls = train_xgb_binary(
        dtrain=d_tr_cls,
        dval=d_va_cls,
        pos=pos,
        neg=neg,
        seed=args.seed,
        max_depth=hp["max_depth"],
        learning_rate=hp["learning_rate"],
        num_boost_round=hp["num_boost_round"],
        early_stopping_rounds=hp["early_stopping_rounds"],
        verbose_eval=50,
    )

    art_cls = TrainArtifacts(
        booster=booster_cls,
        impute_stats=stats,
        num_cols=num_cols,
        cat_cols=cat_cols,
        cat_categories=cat_categories,
        target_col=target_col,
        feature_names=feature_names,
        do_scale=model_cfg["do_scale"],
    )
    cls_path = paths.models_dir / f"xgb_{run_id}.joblib"
    save_artifacts(art_cls, cls_path)
    logger.info(f"saved classifier artifacts: {cls_path}")

    # ── regressor branch ───────────────────────────────────────────────
    if reg_target not in train_df.columns:
        raise KeyError(
            f"Regression target '{reg_target}' not found in train split. "
            "Re-run scripts/run_build_dataset.py after adding the fwd-return label, "
            "or pass --reg_target."
        )

    # Drop rows where the regression target is NaN (tail of each permno's history)
    tr_mask = train_df[reg_target].notna().values
    va_mask = val_df[reg_target].notna().values

    date_col = model_cfg.get("date_col", "DlyCalDt")

    if args.reg_mode == "regression":
        X_tr_reg = train_df.loc[tr_mask, num_cols + cat_cols].copy()
        y_tr_reg = train_df.loc[tr_mask, reg_target].astype("float32").values
        X_va_reg = val_df.loc[va_mask, num_cols + cat_cols].copy()
        y_va_reg = val_df.loc[va_mask, reg_target].astype("float32").values

        d_tr_reg = build_dmatrix(X_tr_reg, y_tr_reg, enable_categorical=True, feature_names=feature_names)
        d_va_reg = build_dmatrix(X_va_reg, y_va_reg, enable_categorical=True, feature_names=feature_names)

        logger.info(
            "training regressor: target=%s, n_train=%d, n_val=%d, y_mean=%.4e, y_std=%.4e",
            reg_target, len(y_tr_reg), len(y_va_reg),
            float(np.mean(y_tr_reg)), float(np.std(y_tr_reg)),
        )
        booster_reg = train_xgb_regressor(
            dtrain=d_tr_reg,
            dval=d_va_reg,
            seed=args.seed,
            max_depth=reg_hp["max_depth"],
            learning_rate=reg_hp["learning_rate"],
            num_boost_round=reg_hp["num_boost_round"],
            early_stopping_rounds=reg_hp["early_stopping_rounds"],
            verbose_eval=50,
        )
    else:
        objective = "rank:ndcg" if args.reg_mode == "rank_ndcg" else "rank:pairwise"
        n_bins = int(args.rank_n_bins)

        def _build_rank_dmatrix(df: pd.DataFrame, mask: np.ndarray, tag: str):
            # Sort rows by trading date so each group is contiguous.
            sub = df.loc[mask].copy()
            sub = sub.sort_values(date_col, kind="mergesort").reset_index(drop=True)
            # Per-day quantile bin on forward return -> relevance in {0, ..., n_bins-1}.
            # pd.qcut with duplicates='drop' returns fewer bins on days with many ties —
            # we then re-code to integer codes, treating NaN (all-equal-day) as 0.
            def _bin(series: pd.Series) -> pd.Series:
                try:
                    codes = pd.qcut(series, q=n_bins, labels=False, duplicates="drop")
                    return codes.fillna(0).astype(np.int32)
                except ValueError:
                    return pd.Series(np.zeros(len(series), dtype=np.int32), index=series.index)
            sub["_rel"] = (
                sub.groupby(date_col, observed=True)[reg_target]
                .transform(_bin)
                .astype(np.int32)
            )
            X = sub[num_cols + cat_cols].copy()
            y = sub["_rel"].values.astype(np.int32)
            group_sizes = (
                sub.groupby(date_col, observed=True).size()
                .sort_index()
                .values.astype(np.int64)
            )
            dm = build_dmatrix(X, y, enable_categorical=True, feature_names=feature_names)
            dm.set_group(group_sizes)
            logger.info(
                "[rank-%s] target=%s, n_rows=%d, n_groups=%d, group_size: mean=%.1f min=%d max=%d",
                tag, reg_target, len(y), len(group_sizes),
                float(group_sizes.mean()), int(group_sizes.min()), int(group_sizes.max()),
            )
            return dm

        d_tr_reg = _build_rank_dmatrix(train_df, tr_mask, "train")
        d_va_reg = _build_rank_dmatrix(val_df,   va_mask, "val")

        logger.info(
            "training ranker: objective=%s, n_bins=%d, rounds=%d",
            objective, n_bins, reg_hp["num_boost_round"],
        )
        booster_reg = train_xgb_ranker(
            dtrain=d_tr_reg,
            dval=d_va_reg,
            seed=args.seed,
            max_depth=reg_hp["max_depth"],
            learning_rate=reg_hp["learning_rate"],
            num_boost_round=reg_hp["num_boost_round"],
            early_stopping_rounds=reg_hp["early_stopping_rounds"],
            objective=objective,
            verbose_eval=50,
        )

    art_reg = RegArtifacts(
        booster=booster_reg,
        impute_stats=stats,
        num_cols=num_cols,
        cat_cols=cat_cols,
        cat_categories=cat_categories,
        target_col=reg_target,
        feature_names=feature_names,
        do_scale=model_cfg["do_scale"],
    )
    reg_path = paths.models_dir / f"xgb_reg_{run_id}.joblib"
    save_artifacts(art_reg, reg_path)
    logger.info(f"saved regressor artifacts: {reg_path}")

    meta_lines = [
        f"run_id={run_id}",
        f"cls_artifacts={cls_path}",
        f"reg_artifacts={reg_path}",
        f"cls_target={target_col}",
        f"reg_target={reg_target}",
    ]
    (run_dir / "meta.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")
    logger.info(f"run_dir: {run_dir}")


if __name__ == "__main__":
    main()
