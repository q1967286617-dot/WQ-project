"""
Week 17 实验一：LightGBM LambdaRank 排序模型

用每日截面 7 日前向收益率分位作为 relevance score，训练 LambdaRank，
生成测试集 scores 并写入 preds/test_preds.parquet（与 XGBoost 格式兼容）。

Usage:
    python scripts/run_train_lambdarank.py --run_id lgbr_s42 --seed 42
    python scripts/run_train_lambdarank.py --run_id lgbr_s42 --seed 42 --skip_backtest
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.modeling.preprocess import (
    apply_imputer_and_scaler,
    fit_train_imputer_and_scaler,
    prepare_categorical,
    split_xy,
)
from src.utils.paths import ensure_dir, load_yaml, resolve_paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOLDING_TD = 7      # 与回测持仓期一致
N_QUINTILES = 5     # relevance 分 5 档


def _compute_fwd_ret(price: pd.DataFrame, holding_td: int) -> pd.DataFrame:
    """从 DlyOpen 计算 holding_td 日前向收益率（t+1 开盘买入，t+1+H 开盘卖出）。"""
    price = price[["PERMNO", "DlyCalDt", "DlyOpen"]].copy()
    price["DlyCalDt"] = pd.to_datetime(price["DlyCalDt"])
    price = price.sort_values(["PERMNO", "DlyCalDt"])

    # 下一日开盘（入场价）
    price["open_entry"] = price.groupby("PERMNO")["DlyOpen"].shift(-1)
    # 持仓期后开盘（出场价）
    price["open_exit"] = price.groupby("PERMNO")["DlyOpen"].shift(-(1 + holding_td))
    price["fwd_ret"] = price["open_exit"] / price["open_entry"] - 1
    return price[["PERMNO", "DlyCalDt", "fwd_ret"]].dropna()


def _add_quintile_label(df: pd.DataFrame) -> pd.DataFrame:
    """每日截面内对 fwd_ret 做 5 分位，返回 0-4 整数 relevance。"""
    df = df.copy()
    def _qcut(g):
        try:
            g["relevance"] = pd.qcut(g["fwd_ret"], q=N_QUINTILES, labels=False, duplicates="drop")
        except ValueError:
            g["relevance"] = 2  # 极端情况：该日样本太少
        return g
    df = df.groupby("DlyCalDt", group_keys=False).apply(_qcut)
    df["relevance"] = df["relevance"].fillna(2).astype(int)
    return df


def _build_lgb_dataset(panel: pd.DataFrame, feature_names: list[str],
                        label_col: str = "relevance") -> tuple[lgb.Dataset, np.ndarray]:
    """构建 LightGBM ranking dataset，group 按日聚合。"""
    panel = panel.sort_values("DlyCalDt")
    groups = panel.groupby("DlyCalDt").size().values
    X = panel[feature_names].values
    y = panel[label_col].values
    return lgb.Dataset(X, label=y, group=groups, feature_name=feature_names), groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="lgbr_s42")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--model_cfg", default="configs/model.yaml")
    ap.add_argument("--backtest_cfg", default="configs/backtest_v9_optimized.yaml")
    ap.add_argument("--num_leaves", type=int, default=63)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--n_estimators", type=int, default=500)
    ap.add_argument("--skip_backtest", action="store_true")
    args = ap.parse_args()

    paths_cfg = load_yaml(PROJECT_ROOT / args.paths)
    model_cfg = load_yaml(PROJECT_ROOT / args.model_cfg)
    paths = resolve_paths(paths_cfg, project_root=PROJECT_ROOT)

    run_dir = paths.outputs_dir / args.run_id
    preds_dir = run_dir / "preds"
    ensure_dir(preds_dir)

    num_cols = list(model_cfg["num_cols"])
    cat_cols = list(model_cfg["cat_cols"])
    feature_names = num_cols + cat_cols
    target_col = model_cfg.get("target_col", "y_div_10d")

    # ── 1. 载入原始价格数据计算前向收益 ──────────────────────────────────────
    print("Computing forward returns from raw price data...")
    price_raw = pd.read_parquet(PROJECT_ROOT / "data/raw/tableB.parquet",
                                columns=["PERMNO", "DlyCalDt", "DlyOpen"])
    fwd = _compute_fwd_ret(price_raw, HOLDING_TD)
    del price_raw

    # ── 2. 载入特征数据 ───────────────────────────────────────────────────────
    print("Loading feature splits...")
    suffix = "_with_fundamentals"
    train_df = pd.read_parquet(paths.processed_dir / f"train{suffix}.parquet")
    val_df   = pd.read_parquet(paths.processed_dir / f"val{suffix}.parquet")
    test_df  = pd.read_parquet(paths.processed_dir / f"test{suffix}.parquet")

    for df in [train_df, val_df, test_df]:
        df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"])

    # ── 3. 预处理（impute，无需 scale，树模型）────────────────────────────────
    stats = fit_train_imputer_and_scaler(train_df, num_cols)
    train_df = apply_imputer_and_scaler(train_df, num_cols, stats, do_scale=False)
    val_df   = apply_imputer_and_scaler(val_df,   num_cols, stats, do_scale=False)
    test_df  = apply_imputer_and_scaler(test_df,  num_cols, stats, do_scale=False)
    train_df, val_df, test_df = prepare_categorical(train_df, val_df, test_df, cat_cols)

    # 把分类列转为整数编码（LightGBM 原生分类列）
    for col in cat_cols:
        for df in [train_df, val_df, test_df]:
            df[col] = df[col].cat.codes.astype(int)

    # ── 4. 合并前向收益并建立 relevance 标签 ─────────────────────────────────
    print("Building relevance labels...")
    fwd["PERMNO"] = fwd["PERMNO"].astype(int)
    for split_df, name in [(train_df, "train"), (val_df, "val")]:
        split_df["PERMNO"] = split_df["PERMNO"].astype(int)

    train_panel = train_df.merge(fwd, on=["PERMNO", "DlyCalDt"], how="inner")
    val_panel   = val_df.merge(fwd,   on=["PERMNO", "DlyCalDt"], how="inner")
    train_panel = _add_quintile_label(train_panel)
    val_panel   = _add_quintile_label(val_panel)
    print(f"  train rows after fwd merge: {len(train_panel):,}, val: {len(val_panel):,}")

    # ── 5. 训练 LambdaRank ───────────────────────────────────────────────────
    print("Training LightGBM LambdaRank...")
    dtrain, _ = _build_lgb_dataset(train_panel, feature_names)
    dval, _   = _build_lgb_dataset(val_panel,   feature_names)

    params = {
        "objective":       "lambdarank",
        "metric":          "ndcg",
        "ndcg_eval_at":    [5, 10],
        "num_leaves":      args.num_leaves,
        "learning_rate":   args.learning_rate,
        "min_child_samples": 20,
        "subsample":       0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":       1.0,
        "reg_lambda":      1.0,
        "seed":            args.seed,
        "verbose":         -1,
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=30, verbose=False),
        lgb.log_evaluation(period=50),
    ]
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=args.n_estimators,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=callbacks,
    )
    print(f"  Best iteration: {booster.best_iteration}")

    # ── 6. 生成测试集预测 ─────────────────────────────────────────────────────
    print("Generating test predictions...")
    test_df["PERMNO"] = test_df["PERMNO"].astype(int)
    X_test = test_df[feature_names].values
    scores = booster.predict(X_test, num_iteration=booster.best_iteration)

    # 归一化到 [0,1]（与 XGBoost prob 格式对齐）
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        scores = (scores - s_min) / (s_max - s_min)

    # 保存 preds（格式与 XGBoost preds 一致）
    keep_cols = ["DlyCalDt", "PERMNO", target_col,
                 "log_mkt_cap", "turnover_5d", "vol_21d", "SICCD", "has_div_history"]
    preds_df = test_df[[c for c in keep_cols if c in test_df.columns]].copy()
    preds_df = preds_df.rename(columns={"DlyCalDt": "date", "PERMNO": "permno", target_col: "y"})
    preds_df["prob"] = scores

    # industry 列
    if "industry" in test_df.columns:
        preds_df["industry"] = test_df["industry"].values

    out_path = preds_dir / "test_preds.parquet"
    preds_df.to_parquet(out_path, index=False)
    print(f"  Saved predictions → {out_path}")

    # ── 7. 运行回测 ────────────────────────────────────────────────────────────
    if not args.skip_backtest:
        print("Running backtest...")
        cmd = [
            sys.executable, "scripts/run_backtest.py",
            "--run_id", args.run_id,
            "--backtest_cfg", args.backtest_cfg,
            "--split", "test",
        ]
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            print("Backtest stderr:", result.stderr[-2000:])
        else:
            print(result.stdout[-1000:])

    print(f"\nDone. run_id={args.run_id}")


if __name__ == "__main__":
    main()
