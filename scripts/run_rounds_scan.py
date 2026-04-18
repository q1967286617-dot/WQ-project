"""
XGBoost training-round scan for v1 feature set.

Sweeps num_boost_round in [25, 50, 75, 100, 150, 200], trains a fresh model for each,
runs prediction on val + test, and runs a full backtest.  Results are saved as
    outputs/rounds_scan/summary.csv
and printed as a Markdown table at the end.

Usage:
    python scripts/run_rounds_scan.py
    python scripts/run_rounds_scan.py --rounds 25 50 100 200 --split test
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.backtest.benchmark import build_equal_weight_benchmark
from src.backtest.portfolio import simulate_portfolio
from src.backtest.report import _return_metrics, enrich_daily_report
from src.backtest.signal import (
    add_execution_returns,
    add_forward_returns,
    build_backtest_panel,
    compute_stable_gap_cv_threshold,
    run_signal_research,
    prepare_candidate_pool,
    select_top_k_from_pool,
)
from src.experiments.versioning import DEFAULT_VERSION_REGISTRY, load_version_specs, materialize_version_configs
from src.modeling.predict import predict_to_eval_df
from src.modeling.preprocess import (
    apply_imputer_and_scaler,
    fit_train_imputer_and_scaler,
    prepare_categorical,
    split_xy,
)
from src.modeling.train import TrainArtifacts, build_dmatrix, save_artifacts, train_xgb_binary
from src.utils.logging import get_logger
from src.utils.paths import ensure_dir, load_yaml, resolve_paths

logger = get_logger("run_rounds_scan")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V1_NUM_COLS = [
    "log_mkt_cap", "ret_5d", "ret_21d", "vol_5d", "vol_21d", "turnover_5d",
    "turnover_21d", "price_to_high", "volume_spike", "vol_ratio", "turnover_ratio",
    "ret_rel_to_ind", "days_since_last_div", "gap_mean_exp", "gap_med_exp",
    "gap_std_exp", "gap_mean_rN", "gap_std_rN", "gap_cv_exp", "time_to_med_exp",
    "z_to_med_exp", "time_to_mean_rN", "z_to_mean_rN", "div_count_exp",
    "weekday_sin", "weekday_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
]
V1_CAT_COLS = ["quarter", "weekday", "is_month_start", "is_month_end"]
TARGET_COL = "y_div_10d"

KEEP_COLS = [
    "DlyCalDt", "PERMNO", "DlyRet", "DlyPrc", "turnover_5d", "SICCD",
    "industry", "has_div_history", "div_count_exp", "gap_cv_exp", "gap_med_exp",
    "days_since_last_div", "z_to_med_exp", "y_div_10d",
    "DlyOpen", "DlyClose", "DlyHigh", "DlyLow", "DlyBid", "DlyAsk",
]


def _read_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _make_eval_df(df: pd.DataFrame, art: TrainArtifacts, art_path: Path) -> pd.DataFrame:
    return predict_to_eval_df(df, art_path, keep_extra_cols=None)


def _run_backtest_inline(
    preds: pd.DataFrame,
    split_df: pd.DataFrame,
    bt_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Minimal inline backtest; returns key portfolio metrics."""
    panel = build_backtest_panel(preds, split_df)

    # add raw price cols if available in split_df
    raw_price_map = {"DlyOpen": "DlyOpen", "DlyClose": "DlyClose",
                     "DlyBid": "DlyBid", "DlyAsk": "DlyAsk"}
    for col in raw_price_map:
        if col in split_df.columns and col not in panel.columns:
            tmp = split_df[["DlyCalDt", "PERMNO", col]].copy()
            tmp = tmp.rename(columns={"DlyCalDt": "date", "PERMNO": "permno"})
            tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
            tmp["permno"] = tmp["permno"].astype(int)
            panel = panel.merge(tmp, on=["date", "permno"], how="left")

    panel = add_execution_returns(panel)
    panel = add_forward_returns(panel, horizons=[1, 5, 10])

    # --- signal research (val-based policy not available here, use current split) ---
    high_prob_threshold = float(bt_cfg.get("stable_prob_threshold", 0.45))
    _, research_decision = run_signal_research(panel, high_prob_threshold=high_prob_threshold)

    dividend_rules_mode = str(bt_cfg.get("dividend_rules_mode", "auto")).lower()
    if dividend_rules_mode == "true":
        use_div_rules = True
    elif dividend_rules_mode == "false":
        use_div_rules = False
    else:
        use_div_rules = research_decision.use_dividend_rules

    stable_gap_cv_threshold = compute_stable_gap_cv_threshold(
        panel,
        stable_div_count_min=int(bt_cfg.get("stable_div_count_min", 4)),
        quantile=float(bt_cfg.get("stable_gap_cv_quantile", 0.5)),
    )

    pool = prepare_candidate_pool(
        panel=panel,
        stable_gap_cv_threshold=stable_gap_cv_threshold,
        turnover_quantile_min=float(bt_cfg.get("turnover_quantile_min", 0.2)),
        exclude_div_count_le=int(bt_cfg.get("exclude_div_count_le", 1)),
        min_price=float(bt_cfg.get("min_price", 3.0)),
        stable_div_count_min=int(bt_cfg.get("stable_div_count_min", 4)),
        stable_prob_threshold=float(bt_cfg.get("stable_prob_threshold", 0.45)),
        regular_prob_threshold=float(bt_cfg.get("regular_prob_threshold", 0.55)),
        use_dividend_rules=use_div_rules,
    )
    candidates = select_top_k_from_pool(
        pool=pool,
        top_k=int(bt_cfg.get("top_k", 20)),
        max_industry_weight=float(bt_cfg.get("max_industry_weight", 0.25)),
        ranking_mode="prob",
    )

    daily_df, trades_df, _ = simulate_portfolio(
        panel=panel,
        candidates=candidates,
        top_k=int(bt_cfg.get("top_k", 20)),
        holding_td=int(bt_cfg.get("holding_td", 10)),
        cooldown_td=int(bt_cfg.get("cooldown_td", 0)),
        cost_bps_one_way=float(bt_cfg.get("cost_bps_one_way", 10.0)),
        max_industry_weight=float(bt_cfg.get("max_industry_weight", 0.25)),
        ret_col="exec_ret_1d",
        use_bid_ask_spread=bool(bt_cfg.get("use_bid_ask_spread", True)),
        spread_cost_cap_bps_one_way=float(bt_cfg.get("spread_cost_cap_bps_one_way", 100.0)),
    )

    # benchmark
    bm_df = build_equal_weight_benchmark(
        panel=panel,
        min_price=float(bt_cfg.get("min_price", 3.0)),
        ret_col="exec_ret_1d",
        cost_bps_one_way=0.0,
    )
    if not daily_df.empty and not bm_df.empty:
        daily_df = enrich_daily_report(daily_df, bm_df)
        port_metrics = _return_metrics(daily_df["portfolio_ret"])
        excess_metrics = _return_metrics(daily_df["excess_ret"])
    else:
        port_metrics = {"annualized_return": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}
        excess_metrics = {"annualized_return": np.nan}

    return {
        "ann_return": port_metrics.get("annualized_return", np.nan),
        "sharpe": port_metrics.get("sharpe", np.nan),
        "max_drawdown": port_metrics.get("max_drawdown", np.nan),
        "excess_ann": excess_metrics.get("annualized_return", np.nan),
        "use_div_rules": use_div_rules,
        "support_votes": research_decision.support_votes,
        "n_trades": int(len(trades_df)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Scan XGBoost round counts on v1 feature set.")
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--backtest_cfg", default="configs/backtest.yaml")
    ap.add_argument("--rounds", nargs="+", type=int, default=[25, 50, 75, 100, 150, 200],
                    help="List of num_boost_round values to scan.")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--early_stopping_rounds", type=int, default=20)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--max_depth", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    paths_cfg = load_yaml(PROJECT_ROOT / args.paths)
    bt_cfg = load_yaml(PROJECT_ROOT / args.backtest_cfg)
    paths = resolve_paths(paths_cfg, project_root=PROJECT_ROOT)

    out_dir = paths.outputs_dir / "rounds_scan"
    ensure_dir(out_dir)
    ensure_dir(paths.models_dir)

    logger.info("Loading data splits …")
    train_df = _read_df(paths.processed_dir / "train.parquet")
    val_df = _read_df(paths.processed_dir / "val.parquet")
    test_df = _read_df(paths.processed_dir / "test.parquet")
    split_df = val_df if args.split == "val" else test_df

    # --- preprocessing (shared across all rounds) ---
    logger.info("Fitting preprocessing on train split …")
    stats = fit_train_imputer_and_scaler(train_df, V1_NUM_COLS)
    # XGBoost does NOT need scaling
    train_p = apply_imputer_and_scaler(train_df, V1_NUM_COLS, stats, do_scale=False)
    val_p = apply_imputer_and_scaler(val_df, V1_NUM_COLS, stats, do_scale=False)
    test_p = apply_imputer_and_scaler(test_df, V1_NUM_COLS, stats, do_scale=False)

    train_p, val_p, test_p = prepare_categorical(train_p, val_p, test_p, V1_CAT_COLS)
    cat_categories = {c: train_p[c].cat.categories.astype(str).tolist() for c in V1_CAT_COLS}
    feature_names = V1_NUM_COLS + V1_CAT_COLS

    X_train, y_train = split_xy(train_p, V1_NUM_COLS, V1_CAT_COLS, TARGET_COL)
    X_val, y_val = split_xy(val_p, V1_NUM_COLS, V1_CAT_COLS, TARGET_COL)
    X_split, y_split = split_xy(
        val_p if args.split == "val" else test_p,
        V1_NUM_COLS, V1_CAT_COLS, TARGET_COL,
    )

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    dtrain = build_dmatrix(X_train, y_train, enable_categorical=True, feature_names=feature_names)
    dval = build_dmatrix(X_val, y_val, enable_categorical=True, feature_names=feature_names)

    # bt_cfg override: v1 is auto-policy
    bt_cfg_v1 = dict(bt_cfg)
    bt_cfg_v1["dividend_rules_mode"] = "auto"

    results: List[Dict[str, Any]] = []

    for n_rounds in sorted(args.rounds):
        logger.info(f"=== Training XGBoost with num_boost_round={n_rounds} ===")
        np.random.seed(args.seed)

        booster = train_xgb_binary(
            dtrain=dtrain,
            dval=dval,
            pos=pos,
            neg=neg,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            num_boost_round=n_rounds,
            early_stopping_rounds=args.early_stopping_rounds,
            verbose_eval=0,  # silent during scan
        )
        best_iter = getattr(booster, "best_iteration", n_rounds)

        art = TrainArtifacts(
            booster=booster,
            impute_stats=stats,
            num_cols=V1_NUM_COLS,
            cat_cols=V1_CAT_COLS,
            cat_categories=cat_categories,
            target_col=TARGET_COL,
            feature_names=feature_names,
            do_scale=False,
        )
        art_path = paths.models_dir / f"xgb_rounds_scan_{n_rounds}.joblib"
        save_artifacts(art, art_path)

        # --- prediction metrics ---
        val_preds = predict_to_eval_df(val_df, art_path)
        val_aucpr = float(average_precision_score(val_preds["y"], val_preds["prob"]))
        val_auc = float(roc_auc_score(val_preds["y"], val_preds["prob"]))

        split_preds = predict_to_eval_df(split_df, art_path)
        split_aucpr = float(average_precision_score(split_preds["y"], split_preds["prob"]))
        split_auc = float(roc_auc_score(split_preds["y"], split_preds["prob"]))

        logger.info(
            f"  val_aucpr={val_aucpr:.4f} | {args.split}_aucpr={split_aucpr:.4f} | best_iter={best_iter}"
        )

        # --- backtest ---
        bt = _run_backtest_inline(split_preds, split_df, bt_cfg_v1)
        logger.info(
            f"  ann={bt['ann_return']:.2%} | excess={bt['excess_ann']:.2%} | "
            f"sharpe={bt['sharpe']:.4f} | rules={bt['use_div_rules']} (votes={bt['support_votes']})"
        )

        results.append({
            "num_boost_round": n_rounds,
            "best_iteration": best_iter,
            "val_AUPRC": round(val_aucpr * 100, 2),
            "val_AUC": round(val_auc * 100, 2),
            f"{args.split}_AUPRC": round(split_aucpr * 100, 2),
            f"{args.split}_AUC": round(split_auc * 100, 2),
            "ann_return_%": round(bt["ann_return"] * 100, 2),
            "excess_ann_%": round(bt["excess_ann"] * 100, 2),
            "sharpe": round(bt["sharpe"], 4),
            "max_drawdown_%": round(bt["max_drawdown"] * 100, 2),
            "use_div_rules": bt["use_div_rules"],
            "policy_votes": bt["support_votes"],
            "n_trades": bt["n_trades"],
        })

    summary_df = pd.DataFrame(results)
    summary_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"\nSummary saved to: {summary_path}")

    # pretty print
    print("\n=== XGBoost Rounds Scan — v1 feature set ===\n")
    print(summary_df.to_markdown(index=False, floatfmt=".2f"))
    print()

    # Save JSON too
    (out_dir / "summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
