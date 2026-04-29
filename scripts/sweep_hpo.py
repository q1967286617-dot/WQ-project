"""
HPO sweep for v9 (Compustat full features).

Grid: max_depth × learning_rate × subsample (27 combos)
Primary metric: val_aucpr (fast, no backtest needed per combo)
Secondary: runs backtest for top-k configs with optimized params (top_k=15, holding_td=7)

Usage:
    python scripts/sweep_hpo.py
    python scripts/sweep_hpo.py --backtest_top_k 3
    python scripts/sweep_hpo.py --backtest_cfg configs/backtest_v9_optimized.yaml --backtest_top_k 5
"""
from __future__ import annotations

import argparse
import itertools
import sys
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.paths import load_yaml, resolve_paths, ensure_dir
from src.utils.logging import get_logger
from src.modeling.preprocess import (
    fit_train_imputer_and_scaler,
    apply_imputer_and_scaler,
    prepare_categorical,
    split_xy,
)
from src.modeling.train import build_dmatrix, train_xgb_binary
from src.experiments.versioning import load_version_specs

logger = get_logger("sweep_hpo")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Grid ─────────────────────────────────────────────────────────────────────
GRID = {
    "max_depth":     [4, 5, 6],
    "learning_rate": [0.03, 0.05, 0.08],
    "subsample":     [0.7, 0.8, 0.9],
}

FIXED = {
    "colsample_bytree":  0.8,
    "reg_lambda":        1.0,
    "reg_alpha":         1.0,
    "min_child_weight":  1.0,
    "num_boost_round":   200,
    "early_stopping_rounds": 20,
}


def _read_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _run_backtest(run_id: str, paths_arg: str, backtest_cfg: str) -> dict[str, Any]:
    """Run backtest subprocess and return summary dict."""
    cmd = [
        "python", "scripts/run_backtest.py",
        "--paths", paths_arg,
        "--backtest_cfg", backtest_cfg,
        "--run_id", run_id,
        "--split", "test",
    ]
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.warning(f"backtest failed for {run_id}: {proc.stderr[-500:]}")
        return {}
    import json
    summary_path = PROJECT_ROOT / "outputs" / "runs" / run_id / "backtest" / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v9")
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--model_cfg", default="configs/model.yaml")
    ap.add_argument("--version_registry", default="configs/version_registry.yaml")
    ap.add_argument("--backtest_cfg", default="configs/backtest_v9_optimized.yaml",
                    help="Backtest config used for top-k combo evaluation")
    ap.add_argument("--backtest_top_k", type=int, default=0,
                    help="Run full backtest for top-k combos by val_aucpr (0 = skip)")
    ap.add_argument("--out_csv", default="outputs/hpo_results.csv")
    args = ap.parse_args()

    paths_cfg = load_yaml(PROJECT_ROOT / args.paths)
    paths = resolve_paths(paths_cfg, project_root=PROJECT_ROOT)

    specs = load_version_specs(PROJECT_ROOT / args.version_registry)
    spec = specs[args.version]

    suffix = spec.data_suffix  # "_with_fundamentals" for v9
    logger.info(f"Loading splits with suffix='{suffix}'")
    train_df = _read_df(paths.processed_dir / f"train{suffix}.parquet")
    val_df   = _read_df(paths.processed_dir / f"val{suffix}.parquet")

    num_cols = list(spec.num_cols)
    cat_cols = list(spec.cat_cols)
    target_col = "y_div_10d"
    feature_names = num_cols + cat_cols

    # Preprocessing (fit on train)
    stats = fit_train_imputer_and_scaler(train_df, num_cols)
    base_model_cfg = load_yaml(PROJECT_ROOT / args.model_cfg)
    do_scale = base_model_cfg.get("do_scale", True)

    train_df = apply_imputer_and_scaler(train_df, num_cols, stats, do_scale=do_scale)
    val_df   = apply_imputer_and_scaler(val_df,   num_cols, stats, do_scale=do_scale)

    train_df, val_df, _ = prepare_categorical(train_df, val_df, val_df.copy(), cat_cols)

    X_train, y_train = split_xy(train_df, num_cols, cat_cols, target_col)
    X_val,   y_val   = split_xy(val_df,   num_cols, cat_cols, target_col)

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)

    dval = build_dmatrix(X_val, y_val, enable_categorical=True, feature_names=feature_names)

    combos = list(itertools.product(*GRID.values()))
    keys = list(GRID.keys())
    total = len(combos)
    logger.info(f"Starting HPO sweep: {total} combos")

    results: list[dict] = []
    for i, values in enumerate(combos):
        params = dict(zip(keys, values))
        label = ", ".join(f"{k}={v}" for k, v in params.items())
        logger.info(f"[{i+1}/{total}] {label}")

        dtrain = build_dmatrix(X_train, y_train, enable_categorical=True, feature_names=feature_names)
        t0 = time.time()
        booster = train_xgb_binary(
            dtrain=dtrain,
            dval=dval,
            pos=pos,
            neg=neg,
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=FIXED["colsample_bytree"],
            reg_lambda=FIXED["reg_lambda"],
            reg_alpha=FIXED["reg_alpha"],
            min_child_weight=FIXED["min_child_weight"],
            num_boost_round=FIXED["num_boost_round"],
            early_stopping_rounds=FIXED["early_stopping_rounds"],
            verbose_eval=999,  # silent
        )
        elapsed = time.time() - t0

        val_probs = booster.predict(dval)
        aucpr = float(average_precision_score(y_val, val_probs))
        best_iter = int(getattr(booster, "best_iteration", FIXED["num_boost_round"]))

        row = {
            "max_depth": params["max_depth"],
            "learning_rate": params["learning_rate"],
            "subsample": params["subsample"],
            "val_aucpr": aucpr,
            "best_iteration": best_iter,
            "train_sec": round(elapsed, 1),
        }
        results.append(row)
        logger.info(f"  → val_aucpr={aucpr:.4f}  best_iter={best_iter}  ({elapsed:.1f}s)")

    df = pd.DataFrame(results).sort_values("val_aucpr", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    out_path = PROJECT_ROOT / args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 72)
    print(f"HPO SWEEP RESULTS  (version={args.version}, {total} combos)")
    print("=" * 72)
    print(df[["rank", "max_depth", "learning_rate", "subsample", "val_aucpr", "best_iteration"]].to_string(index=False))
    print(f"\nFull results saved → {out_path}")

    if args.backtest_top_k > 0:
        top_rows = df.head(args.backtest_top_k)
        print(f"\nRunning backtest for top {args.backtest_top_k} combos...")
        import joblib, json
        from src.modeling.train import TrainArtifacts, save_artifacts
        from src.modeling.preprocess import ImputeScaleStats

        backtest_results: list[dict] = []
        for _, row in top_rows.iterrows():
            run_id = (
                f"hpo_{args.version}"
                f"_d{int(row['max_depth'])}"
                f"_lr{str(row['learning_rate']).replace('.','')}"
                f"_sub{str(row['subsample']).replace('.','')}"
            )
            art_path = paths.models_dir / f"xgb_{run_id}.joblib"
            preds_dir = PROJECT_ROOT / "outputs" / "runs" / run_id / "preds"
            ensure_dir(preds_dir)

            # Retrain with this combo (fast since we cache artifact)
            if not art_path.exists():
                dtrain = build_dmatrix(X_train, y_train, enable_categorical=True, feature_names=feature_names)
                booster = train_xgb_binary(
                    dtrain=dtrain, dval=dval, pos=pos, neg=neg,
                    max_depth=int(row["max_depth"]),
                    learning_rate=float(row["learning_rate"]),
                    subsample=float(row["subsample"]),
                    colsample_bytree=FIXED["colsample_bytree"],
                    reg_lambda=FIXED["reg_lambda"],
                    reg_alpha=FIXED["reg_alpha"],
                    min_child_weight=FIXED["min_child_weight"],
                    num_boost_round=FIXED["num_boost_round"],
                    early_stopping_rounds=FIXED["early_stopping_rounds"],
                    verbose_eval=999,
                )
                cat_categories = {c: train_df[c].cat.categories.astype(str).tolist() for c in cat_cols}
                art = TrainArtifacts(
                    booster=booster, impute_stats=stats,
                    num_cols=num_cols, cat_cols=cat_cols,
                    cat_categories=cat_categories, target_col=target_col,
                    feature_names=feature_names, do_scale=do_scale,
                )
                save_artifacts(art, art_path)

            # Generate test predictions
            test_preds_path = preds_dir / "test_preds.parquet"
            if not test_preds_path.exists():
                cmd = ["python", "scripts/run_predict.py",
                       "--run_id", run_id, "--split", "test",
                       "--data_suffix", suffix]
                proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
                if proc.returncode != 0:
                    logger.warning(f"predict failed: {proc.stderr[-300:]}")
                    continue

            bt = _run_backtest(run_id, args.paths, args.backtest_cfg)
            sharpe = bt.get("portfolio", {}).get("sharpe", float("nan"))
            ann_ret = bt.get("portfolio", {}).get("annualized_return", float("nan"))
            excess_ann = bt.get("excess_vs_benchmark", {}).get("annualized_return", float("nan"))
            backtest_results.append({
                "run_id": run_id,
                "max_depth": int(row["max_depth"]),
                "learning_rate": float(row["learning_rate"]),
                "subsample": float(row["subsample"]),
                "val_aucpr": float(row["val_aucpr"]),
                "sharpe": sharpe,
                "ann_return": ann_ret,
                "excess_ann": excess_ann,
            })

        if backtest_results:
            bt_df = pd.DataFrame(backtest_results).sort_values("sharpe", ascending=False)
            print("\n" + "=" * 72)
            print("BACKTEST RESULTS FOR TOP COMBOS")
            print("=" * 72)
            print(bt_df[["max_depth", "learning_rate", "subsample",
                          "val_aucpr", "sharpe", "ann_return", "excess_ann"]].to_string(index=False))
            bt_out = PROJECT_ROOT / args.out_csv.replace(".csv", "_backtest.csv")
            bt_df.to_csv(bt_out, index=False)
            print(f"\nBacktest results saved → {bt_out}")


if __name__ == "__main__":
    main()
