"""
Stage-2 LambdaRank Reranker
============================
Trains a LightGBM LambdaRank model that ranks stocks WITHIN the v12 candidate
pool, using **net** forward-return quantile as relevance.

Key differences from lgbr_* (which replaced v12 entirely):
  - Pool filtering: still done by the v12 classifier (prob >= threshold).
  - Reranker: only responsible for ranking within the already-filtered pool.
  - Relevance: quantile(fwd_ret_10d - est_cost) per day, not raw fwd_ret.
  - Training subset: optionally restrict to pool-like rows only.

Usage:
    python scripts/run_train_reranker.py --run_id rerank_v1 --cls_run v12_run1
    python scripts/run_train_reranker.py --run_id rerank_v1 --cls_run v12_run1 \\
        --pool_only --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.paths import ensure_dir, load_yaml, resolve_paths
from src.modeling.preprocess import (
    apply_imputer_and_scaler,
    fit_train_imputer_and_scaler,
    prepare_categorical,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COST_ONE_WAY_BPS = 10.0   # same as backtest cost assumption
N_BINS = 5                 # relevance levels 0-4


def _net_ret_quantile(df: pd.DataFrame, fwd_col: str, cost: float, n_bins: int) -> pd.Series:
    """Per-day quantile bin of (fwd_ret - round_trip_cost), 0 = worst, n_bins-1 = best."""
    net = df[fwd_col] - 2.0 * cost
    def _qcut(g):
        try:
            return pd.qcut(g, q=n_bins, labels=False, duplicates="drop").astype(float)
        except Exception:
            return pd.Series(float(n_bins // 2), index=g.index)
    return df.groupby("DlyCalDt")[fwd_col].transform(
        lambda s: _qcut(s - 2.0 * cost)
    ).fillna(n_bins // 2).astype(int)


def _encode_cats(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """Encode categorical columns as integer codes for LightGBM."""
    df = df.copy()
    for c in cat_cols:
        if c in df.columns:
            df[c] = pd.Categorical(df[c]).codes.astype(np.float32)
    return df


def _build_dataset(df: pd.DataFrame, feature_names: list, cat_cols: list) -> lgb.Dataset:
    df = df.sort_values("DlyCalDt")
    df = _encode_cats(df, cat_cols)
    groups = df.groupby("DlyCalDt").size().values
    X = df[feature_names].values.astype(np.float32)
    y = df["relevance"].values.astype(np.int32)
    cat_feature_indices = [feature_names.index(c) for c in cat_cols if c in feature_names]
    return lgb.Dataset(X, label=y, group=groups, feature_name=feature_names,
                       categorical_feature=cat_feature_indices, free_raw_data=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id",   default="rerank_v1")
    ap.add_argument("--cls_run",  default="v12_run1",
                    help="Run-id whose val/test_preds.parquet provides the pool "
                         "membership signal (prob column).")
    ap.add_argument("--paths",    default="configs/paths.yaml")
    ap.add_argument("--model_cfg",default="configs/model.yaml")
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--pool_only", action="store_true",
                    help="Restrict training to pool-like rows "
                         "(prob >= pool_prob_min from cls_run val preds).")
    ap.add_argument("--pool_prob_min", type=float, default=0.45,
                    help="Prob threshold for pool membership in training data.")
    ap.add_argument("--n_bins",   type=int, default=N_BINS)
    ap.add_argument("--n_leaves", type=int, default=31)
    ap.add_argument("--lr",       type=float, default=0.05)
    ap.add_argument("--n_rounds", type=int, default=300)
    ap.add_argument("--early_stopping", type=int, default=30)
    args = ap.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    model_cfg  = load_yaml(Path(args.model_cfg))
    paths = resolve_paths(paths_cfg, project_root=Path.cwd())

    out_dir = paths.outputs_dir / args.run_id
    ensure_dir(out_dir / "preds")

    # ── load processed splits ───────────────────────────────────────────────
    print("Loading splits...")
    train_df = pd.read_parquet(paths.processed_dir / "train.parquet")
    val_df   = pd.read_parquet(paths.processed_dir / "val.parquet")
    test_df  = pd.read_parquet(paths.processed_dir / "test.parquet")

    for df in (train_df, val_df, test_df):
        df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"])

    fwd_col = f"fwd_ret_{model_cfg.get('H_label', 10) or 10}d"
    if fwd_col not in train_df.columns:
        raise KeyError(f"'{fwd_col}' not in train.parquet — rebuild dataset first.")

    num_cols  = list(model_cfg["num_cols"])
    cat_cols  = list(model_cfg["cat_cols"])
    feature_names = num_cols + cat_cols

    # ── preprocessing (same pipeline as classifier) ─────────────────────────
    print("Preprocessing...")
    stats = fit_train_imputer_and_scaler(train_df, num_cols)
    train_df = apply_imputer_and_scaler(train_df, num_cols, stats, do_scale=False)
    val_df   = apply_imputer_and_scaler(val_df,   num_cols, stats, do_scale=False)
    test_df  = apply_imputer_and_scaler(test_df,  num_cols, stats, do_scale=False)
    train_df, val_df, test_df = prepare_categorical(train_df, val_df, test_df, cat_cols)

    # ── net-return relevance labels ─────────────────────────────────────────
    cost = COST_ONE_WAY_BPS / 10_000.0
    print(f"Computing net-return relevance (cost={cost*10000:.0f}bps one-way)...")
    train_df["relevance"] = _net_ret_quantile(train_df, fwd_col, cost, args.n_bins)
    val_df["relevance"]   = _net_ret_quantile(val_df,   fwd_col, cost, args.n_bins)

    # ── optional: restrict training to pool-like rows ───────────────────────
    if args.pool_only:
        val_preds_path = paths.outputs_dir / args.cls_run / "preds" / "val_preds.parquet"
        if not val_preds_path.exists():
            print(f"WARNING: {val_preds_path} not found; skipping pool filter.")
        else:
            val_preds = pd.read_parquet(val_preds_path, columns=["permno", "date", "prob"])
            val_preds["date"] = pd.to_datetime(val_preds["date"])
            val_preds["permno"] = val_preds["permno"].astype(int)
            # pool membership proxy for val
            val_df["_permno"] = val_df["PERMNO"].astype(int)
            val_df["_date"]   = val_df["DlyCalDt"]
            val_df = val_df.merge(
                val_preds.rename(columns={"prob": "_cls_prob"}),
                left_on=["_permno", "_date"], right_on=["permno", "date"], how="left"
            )
            before = len(val_df)
            val_df = val_df[val_df["_cls_prob"].fillna(0) >= args.pool_prob_min]
            print(f"  val pool filter: {before} → {len(val_df)} rows "
                  f"(prob >= {args.pool_prob_min})")
            # For train: use signal-group proxy (div_count_exp >= 1 as rough filter)
            if "div_count_exp" in train_df.columns:
                before = len(train_df)
                train_df = train_df[train_df["div_count_exp"].fillna(0) >= 1]
                print(f"  train pool proxy: {before} → {len(train_df)} rows "
                      f"(div_count_exp >= 1)")

    # Drop rows with missing fwd_ret
    train_df = train_df.dropna(subset=[fwd_col])
    val_df   = val_df.dropna(subset=[fwd_col])
    print(f"Training rows: {len(train_df):,}  |  Val rows: {len(val_df):,}")

    # ── build LGB datasets ──────────────────────────────────────────────────
    ds_train = _build_dataset(train_df, feature_names, cat_cols)
    ds_val   = _build_dataset(val_df,   feature_names, cat_cols)

    params = {
        "objective":       "lambdarank",
        "metric":          "ndcg",
        "ndcg_eval_at":    [5, 10, 15],
        "num_leaves":      args.n_leaves,
        "learning_rate":   args.lr,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":    5,
        "reg_lambda":      1.0,
        "seed":            args.seed,
        "verbose":         -1,
        "n_jobs":          -1,
    }

    callbacks = [
        lgb.early_stopping(args.early_stopping, verbose=True),
        lgb.log_evaluation(50),
    ]

    print(f"Training LambdaRank reranker (n_rounds={args.n_rounds})...")
    booster = lgb.train(
        params,
        ds_train,
        num_boost_round=args.n_rounds,
        valid_sets=[ds_val],
        valid_names=["val"],
        callbacks=callbacks,
    )

    model_path = paths.models_dir / f"reranker_{args.run_id}.lgb"
    booster.save_model(str(model_path))
    print(f"Saved reranker model: {model_path}")

    # ── generate predictions for val and test ───────────────────────────────
    def _score_and_save(df: pd.DataFrame, cls_preds_path: Path, split: str):
        """Score df with reranker; merge with cls prob; save combined preds."""
        if not cls_preds_path.exists():
            print(f"  {split}: cls preds not found at {cls_preds_path}, skipping.")
            return

        cls_preds = pd.read_parquet(cls_preds_path)
        cls_preds["date"]   = pd.to_datetime(cls_preds["date"])
        cls_preds["permno"] = cls_preds["permno"].astype(int)

        df = df.copy()
        df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"])
        df["PERMNO"]   = df["PERMNO"].astype(int)

        df_scored = apply_imputer_and_scaler(df, num_cols, stats, do_scale=False)
        df_enc = _encode_cats(df_scored, cat_cols)
        X = df_enc[feature_names].fillna(0).values.astype(np.float32)
        df["score_rerank"] = booster.predict(X)

        merged = cls_preds.merge(
            df[["PERMNO", "DlyCalDt", "score_rerank"]].rename(
                columns={"PERMNO": "permno", "DlyCalDt": "date"}
            ),
            on=["permno", "date"], how="left"
        )
        # Expose as score_reg so ranking_mode=return_score in signal.py can read it
        merged["score_reg"] = merged["score_rerank"]

        out_path = out_dir / "preds" / f"{split}_preds.parquet"
        merged.to_parquet(out_path, index=False)
        print(f"  saved {split} preds: {out_path}  (rows={len(merged)})")

    cls_run_dir = paths.outputs_dir / args.cls_run
    print("Generating predictions...")
    _score_and_save(val_df  if not args.pool_only else pd.read_parquet(paths.processed_dir / "val.parquet"),
                    cls_run_dir / "preds" / "val_preds.parquet", "val")
    _score_and_save(test_df,
                    cls_run_dir / "preds" / "test_preds.parquet", "test")

    # ── quick IC check ──────────────────────────────────────────────────────
    print("\n── IC check (test set) ──")
    test_raw = pd.read_parquet(paths.processed_dir / "test.parquet")
    test_raw["DlyCalDt"] = pd.to_datetime(test_raw["DlyCalDt"])
    test_raw["PERMNO"]   = test_raw["PERMNO"].astype(int)
    test_enc = _encode_cats(
        apply_imputer_and_scaler(test_raw, num_cols, stats, do_scale=False),
        cat_cols
    )
    test_raw["score_rerank"] = booster.predict(test_enc[feature_names].values.astype(np.float32))
    from scipy.stats import spearmanr
    ics = []
    for _, g in test_raw.dropna(subset=[fwd_col]).groupby("DlyCalDt"):
        if len(g) < 5: continue
        ic, _ = spearmanr(g["score_rerank"], g[fwd_col])
        ics.append(ic)
    arr = np.array(ics)
    print(f"  score_rerank IC: mean={arr.mean():.4f}, std={arr.std():.4f}, "
          f"ICIR={arr.mean()/arr.std():.3f}  (n={len(arr)})")


if __name__ == "__main__":
    main()
