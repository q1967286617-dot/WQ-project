"""
Extended backtest parameter sweep:
  - Fine-grained top_k × holding_td around current best (15, 7)
  - Tradability filter: max bid_ask_spread_5d threshold

Usage:
    python scripts/sweep_backtest_v2.py --run_id v9_run1
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import itertools

import numpy as np
import pandas as pd

from src.backtest.portfolio import simulate_portfolio
from src.backtest.benchmark import build_equal_weight_benchmark
from src.backtest.report import enrich_daily_report, _return_metrics
from src.backtest.signal import (
    add_execution_returns,
    add_forward_returns,
    build_backtest_panel,
    compute_stable_gap_cv_threshold,
    infer_execution_basis,
    merge_execution_price_data,
    prepare_candidate_pool,
    select_top_k_from_pool,
)
from src.utils.paths import load_yaml, resolve_paths
from src.utils.logging import get_logger

logger = get_logger("sweep_v2")

RAW_PRICE_COLS = ["PERMNO", "DlyCalDt", "DlyOpen", "DlyClose", "DlyHigh", "DlyLow", "DlyBid", "DlyAsk"]

TOP_K_VALUES   = [10, 12, 13, 14, 15, 16, 17, 18, 20]
HOLDING_VALUES = [6, 7, 8, 9]
SPREAD_CAPS    = [None, 0.0020, 0.0010, 0.0005]   # None = no filter


def _read_df(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _load_price_data(table_b_path: Path, base_df: pd.DataFrame) -> pd.DataFrame:
    permnos  = set(pd.to_numeric(base_df["PERMNO"], errors="coerce").dropna().astype(int).unique())
    min_date = pd.to_datetime(base_df["DlyCalDt"]).min()
    max_date = pd.to_datetime(base_df["DlyCalDt"]).max()
    source   = table_b_path.with_suffix(".parquet") if table_b_path.with_suffix(".parquet").exists() else table_b_path
    if source.suffix == ".parquet":
        raw = pd.read_parquet(source, columns=RAW_PRICE_COLS)
    else:
        parts = []
        for chunk in pd.read_csv(source, usecols=RAW_PRICE_COLS, chunksize=250000):
            chunk["DlyCalDt"] = pd.to_datetime(chunk["DlyCalDt"], errors="coerce")
            chunk["PERMNO"]   = pd.to_numeric(chunk["PERMNO"], errors="coerce").astype("Int64")
            chunk = chunk[chunk["PERMNO"].isin(permnos) & chunk["DlyCalDt"].between(min_date, max_date)]
            if not chunk.empty:
                parts.append(chunk)
        raw = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=RAW_PRICE_COLS)
    raw["DlyCalDt"] = pd.to_datetime(raw["DlyCalDt"], errors="coerce")
    raw["PERMNO"]   = pd.to_numeric(raw["PERMNO"], errors="coerce").astype("Int64")
    raw = raw[raw["PERMNO"].isin(permnos) & raw["DlyCalDt"].between(min_date, max_date)].copy()
    raw = raw.rename(columns={"PERMNO": "permno", "DlyCalDt": "date"})
    raw["permno"] = raw["permno"].astype(int)
    return raw.drop_duplicates(subset=["date", "permno"]).sort_values(["date", "permno"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id",       default="v9_run1")
    ap.add_argument("--split",        default="test")
    ap.add_argument("--paths",        default="configs/paths.yaml")
    ap.add_argument("--backtest_cfg", default="configs/backtest_v9_optimized.yaml")
    args = ap.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    bt_cfg    = load_yaml(Path(args.backtest_cfg))
    paths     = resolve_paths(paths_cfg, project_root=Path.cwd())
    run_dir   = paths.outputs_dir / args.run_id

    # ── load once ─────────────────────────────────────────────────────────────
    preds_path = run_dir / "preds" / f"{args.split}_preds.parquet"
    split_path = paths.processed_dir / f"{args.split}.parquet"
    print(f"Loading: {preds_path.name}")
    preds_df = _read_df(preds_path)
    split_df = _read_df(split_path)

    print("Building panel…")
    base_panel = build_backtest_panel(preds_df, split_df)
    price_df   = _load_price_data(paths.tableB_path, split_df)
    base_panel = merge_execution_price_data(base_panel, price_df)
    base_panel = add_execution_returns(base_panel)
    all_horizons = sorted({1} | set(HOLDING_VALUES))
    base_panel = add_forward_returns(base_panel, horizons=all_horizons)
    ret_col = "exec_ret_1d"
    print(f"Panel rows: {len(base_panel):,}")

    val_df = _read_df(paths.processed_dir / "val.parquet")
    stable_gap_cv_threshold = compute_stable_gap_cv_threshold(
        val_df,
        stable_div_count_min=int(bt_cfg.get("stable_div_count_min", 4)),
        quantile=float(bt_cfg.get("stable_gap_cv_quantile", 0.5)),
    )

    # base candidate pool (no spread filter yet)
    base_pool = prepare_candidate_pool(
        panel=base_panel,
        stable_gap_cv_threshold=stable_gap_cv_threshold,
        turnover_quantile_min=float(bt_cfg["turnover_quantile_min"]),
        exclude_div_count_le=int(bt_cfg["exclude_div_count_le"]),
        min_price=float(bt_cfg["min_price"]),
        stable_div_count_min=int(bt_cfg.get("stable_div_count_min", 4)),
        stable_prob_threshold=float(bt_cfg.get("stable_prob_threshold", 0.45)),
        regular_prob_threshold=float(bt_cfg.get("regular_prob_threshold", 0.55)),
        use_dividend_rules=True,
    )

    # spread percentiles for reference
    spread_col = "bid_ask_spread_5d"
    if spread_col in base_pool.columns:
        eligible_pool = base_pool[base_pool.get("eligible", pd.Series(True, index=base_pool.index)).astype(bool)]
        q50 = eligible_pool[spread_col].quantile(0.50)
        q75 = eligible_pool[spread_col].quantile(0.75)
        q90 = eligible_pool[spread_col].quantile(0.90)
        print(f"bid_ask_spread_5d percentiles in eligible pool: p50={q50:.5f}, p75={q75:.5f}, p90={q90:.5f}")

    # ── sweep ─────────────────────────────────────────────────────────────────
    combos  = list(itertools.product(TOP_K_VALUES, HOLDING_VALUES, SPREAD_CAPS))
    results = []
    print(f"\nSweeping {len(combos)} combinations…\n")

    for top_k, holding_td, spread_cap in combos:
        # apply spread filter
        if spread_cap is not None and spread_col in base_pool.columns:
            pool = base_pool[base_pool[spread_col] <= spread_cap].copy()
        else:
            pool = base_pool.copy()

        candidates = select_top_k_from_pool(
            pool=pool,
            top_k=top_k,
            max_industry_weight=float(bt_cfg.get("max_industry_weight", 1.0)),
            ranking_mode="prob",
        )

        bench = build_equal_weight_benchmark(
            base_panel,
            min_price=float(bt_cfg["min_price"]),
            ret_col=ret_col,
            cost_bps_one_way=float(bt_cfg["cost_bps_one_way"]),
            holding_td=holding_td,
            turnover_quantile_min=float(bt_cfg["turnover_quantile_min"]),
            exclude_div_count_le=int(bt_cfg["exclude_div_count_le"]),
        )

        daily_df, trades_df, _ = simulate_portfolio(
            panel=base_panel,
            candidates=candidates,
            top_k=top_k,
            holding_td=holding_td,
            cooldown_td=int(bt_cfg["cooldown_td"]),
            cost_bps_one_way=float(bt_cfg["cost_bps_one_way"]),
            max_industry_weight=float(bt_cfg.get("max_industry_weight", 1.0)),
            ret_col=ret_col,
            use_bid_ask_spread=bool(bt_cfg.get("use_bid_ask_spread", False)),
            spread_cost_cap_bps_one_way=float(bt_cfg.get("spread_cost_cap_bps_one_way", 100.0)),
        )
        daily_df  = enrich_daily_report(daily_df, bench)
        port_m    = _return_metrics(daily_df["portfolio_ret"])
        excess_m  = _return_metrics(daily_df["portfolio_ret"] - daily_df["benchmark_ret"])

        spread_label = f"{spread_cap:.4f}" if spread_cap else "none"
        results.append({
            "top_k":        top_k,
            "holding_td":   holding_td,
            "spread_cap":   spread_label,
            "sharpe":       round(port_m["sharpe"], 4),
            "ann_return":   round(port_m["annualized_return"] * 100, 2),
            "max_dd":       round(port_m["max_drawdown"] * 100, 2),
            "excess_ann":   round(excess_m["annualized_return"] * 100, 2),
            "excess_sharpe":round(excess_m["sharpe"], 4),
            "n_trades":     len(trades_df),
        })

        if spread_cap is None:
            print(f"  k={top_k:2d} h={holding_td} spread=none | "
                  f"Sharpe={port_m['sharpe']:.4f} | ann={port_m['annualized_return']:.2%} | "
                  f"excess={excess_m['annualized_return']:.2%}")

    df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    out_path = run_dir / "backtest_param_sweep_v2.csv"
    df.to_csv(out_path, index=False)

    print(f"\n{'='*85}")
    print("TOP 15 COMBINATIONS BY SHARPE")
    print(f"{'='*85}")
    print(df.head(15).to_string(index=False))

    print(f"\n--- Best Sharpe per spread_cap ---")
    for cap, grp in df.groupby("spread_cap"):
        best = grp.iloc[0]
        print(f"  spread_cap={cap:7s} | best Sharpe={best['sharpe']:.4f} "
              f"k={int(best['top_k'])} h={int(best['holding_td'])} "
              f"excess={best['excess_ann']:.2f}%")

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
