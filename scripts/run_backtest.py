from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.append(str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.backtest.benchmark import (
    build_equal_weight_benchmark,
    build_random_candidates,
    build_oracle_candidates,
    compute_alpha_capture,
)
from src.backtest.portfolio import simulate_portfolio
from src.backtest.report import (
    build_trade_attribution,
    enrich_daily_report,
    summarize_backtest,
    write_backtest_outputs,
    _return_metrics,
)
from src.backtest.signal import (
    add_execution_returns,
    add_forward_returns,
    build_backtest_panel,
    build_daily_candidates,
    compute_stable_gap_cv_threshold,
    infer_execution_basis,
    merge_execution_price_data,
    run_signal_research,
)
from src.modeling.predict import predict_to_eval_df
from src.utils.logging import get_logger
from src.utils.paths import load_yaml, resolve_paths

logger = get_logger("run_backtest")

KEEP_EXTRA_COLS = ["log_mkt_cap", "turnover_5d", "vol_21d", "SICCD", "industry", "has_div_history"]
RAW_PRICE_COLS  = ["PERMNO", "DlyCalDt", "DlyOpen", "DlyClose", "DlyHigh", "DlyLow", "DlyBid", "DlyAsk"]


# ── 工具函数（与原版完全一致） ────────────────────────────────────────────────

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


def _resolve_preds_path(run_dir: Path, split: str, override: str | None) -> Path:
    if override:
        return Path(override)
    parquet = run_dir / "preds" / f"{split}_preds.parquet"
    csv = parquet.with_suffix(".csv")
    if parquet.exists():
        return parquet
    if csv.exists():
        return csv
    raise FileNotFoundError(f"Missing predictions: {parquet} or {csv}. Run scripts/run_predict.py first.")


def _resolve_artifacts_path(models_dir: Path, run_id: str, override: str | None) -> Path:
    if override:
        return Path(override)
    path = models_dir / f"xgb_{run_id}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifacts: {path}")
    return path


def policy_split_names(target_split: str) -> List[str]:
    if target_split == "test":
        return ["val"]
    return []


def threshold_reference_split_names(target_split: str) -> List[str]:
    order = ["train", "val", "test"]
    idx = order.index(target_split)
    return order[:idx]


def _load_named_splits(processed_dir: Path, split_names: List[str]) -> pd.DataFrame:
    parts = []
    for name in split_names:
        path = processed_dir / f"{name}.parquet"
        if path.exists():
            parts.append(_read_df(path))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _tableb_source_path(table_b_path: Path) -> Path:
    parquet = table_b_path.with_suffix(".parquet")
    return parquet if parquet.exists() else table_b_path


def _load_execution_price_data(table_b_path: Path, base_df: pd.DataFrame) -> pd.DataFrame:
    if base_df.empty:
        return pd.DataFrame(columns=["date", "permno", "DlyOpen", "DlyClose",
                                     "DlyHigh", "DlyLow", "DlyBid", "DlyAsk"])
    source   = _tableb_source_path(table_b_path)
    permnos  = set(pd.to_numeric(base_df["PERMNO"], errors="coerce").dropna().astype(int).unique().tolist())
    min_date = pd.to_datetime(base_df["DlyCalDt"], errors="coerce").min()
    max_date = pd.to_datetime(base_df["DlyCalDt"], errors="coerce").max()

    if source.suffix.lower() == ".parquet":
        raw = pd.read_parquet(source, columns=RAW_PRICE_COLS)
        raw["DlyCalDt"] = pd.to_datetime(raw["DlyCalDt"], errors="coerce")
        raw["PERMNO"]   = pd.to_numeric(raw["PERMNO"], errors="coerce").astype("Int64")
        raw = raw[raw["PERMNO"].isin(permnos) & raw["DlyCalDt"].between(min_date, max_date)].copy()
    else:
        parts = []
        reader = pd.read_csv(source, usecols=RAW_PRICE_COLS, chunksize=250000)
        for chunk in reader:
            chunk["DlyCalDt"] = pd.to_datetime(chunk["DlyCalDt"], errors="coerce")
            chunk["PERMNO"]   = pd.to_numeric(chunk["PERMNO"], errors="coerce").astype("Int64")
            chunk = chunk[chunk["PERMNO"].isin(permnos) & chunk["DlyCalDt"].between(min_date, max_date)]
            if not chunk.empty:
                parts.append(chunk)
        raw = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=RAW_PRICE_COLS)

    raw = raw.rename(columns={"PERMNO": "permno", "DlyCalDt": "date"})
    raw["permno"] = pd.to_numeric(raw["permno"], errors="coerce").astype(int)
    raw["date"]   = pd.to_datetime(raw["date"], errors="coerce")
    return raw.drop_duplicates(subset=["date", "permno"]).sort_values(["date", "permno"]).reset_index(drop=True)


def _prepare_panel(base_preds, base_split, table_b_path, holding_td):
    panel    = build_backtest_panel(base_preds, base_split)
    price_df = _load_execution_price_data(table_b_path, base_split)
    panel    = merge_execution_price_data(panel, price_df)
    panel    = add_execution_returns(panel)
    panel    = add_forward_returns(panel, horizons=(1, 5, int(holding_td)))
    return panel


def _build_research_basis_panel(ref_df, art_path, table_b_path, holding_td):
    ref_preds = predict_to_eval_df(ref_df, art_path, keep_extra_cols=KEEP_EXTRA_COLS)
    return _prepare_panel(ref_preds, ref_df, table_b_path, holding_td)


def _decision_payload(policy_decision, diagnostic_decision, mode,
                      active_value, policy_source, execution_basis) -> Dict:
    return {
        "mode":                               mode,
        "use_dividend_rules":                 bool(active_value),
        "policy_source":                      policy_source,
        "execution_basis":                    execution_basis,
        "research_recommended_dividend_rules": bool(policy_decision.use_dividend_rules),
        "support_votes":                      int(policy_decision.support_votes),
        "total_checks":                       int(policy_decision.total_checks),
        "checks":                             policy_decision.checks,
        "current_split_diagnostic": {
            "recommended_dividend_rules": bool(diagnostic_decision.use_dividend_rules),
            "support_votes":              int(diagnostic_decision.support_votes),
            "total_checks":               int(diagnostic_decision.total_checks),
            "checks":                     diagnostic_decision.checks,
        },
    }


def _cost_model_payload(bt_cfg: Dict, execution_basis: str) -> Dict:
    use_spread = bool(bt_cfg.get("use_bid_ask_spread", False))
    spread_cap = float(bt_cfg.get("spread_cost_cap_bps_one_way", 100.0)) if use_spread else None
    return {
        "execution_basis":           execution_basis,
        "fixed_cost_bps_one_way":    float(bt_cfg.get("cost_bps_one_way", 0.0)),
        "use_bid_ask_spread":        use_spread,
        "spread_cost_model":         "half_spread_from_daily_bid_ask" if use_spread else "disabled",
        "spread_cost_cap_bps_one_way": spread_cap,
    }


# ── 三点坐标系辅助函数 ─────────────────────────────────────────────────────────

def _simulate_reference(
    label: str,
    candidates: pd.DataFrame,
    panel: pd.DataFrame,
    bt_cfg: Dict,
    ret_col: str = "exec_ret_1d",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    用随机或 Oracle 候选集运行 simulate_portfolio。
    成本结构、持仓期、冷却期与主策略完全一致，确保三组可比。
    """
    logger.info(f"simulating {label} (n_candidates={len(candidates)})")
    if candidates.empty:
        logger.warning(f"{label} candidates empty, skipping")
        return pd.DataFrame(), pd.DataFrame()

    daily_df, trades_df, _ = simulate_portfolio(
        panel=panel,
        candidates=candidates,
        top_k=int(bt_cfg["top_k"]),
        holding_td=int(bt_cfg["holding_td"]),
        cooldown_td=int(bt_cfg["cooldown_td"]),
        cost_bps_one_way=float(bt_cfg["cost_bps_one_way"]),
        max_industry_weight=float(bt_cfg.get("max_industry_weight", 1.0)),
        ret_col=ret_col,
        use_bid_ask_spread=bool(bt_cfg.get("use_bid_ask_spread", False)),
        spread_cost_cap_bps_one_way=float(bt_cfg.get("spread_cost_cap_bps_one_way", 100.0)),
    )
    return daily_df, trades_df


def _build_three_way_comparison(
    strategy_daily: pd.DataFrame,
    random_daily: pd.DataFrame,
    oracle_daily: pd.DataFrame,
) -> Dict:
    """
    三点坐标系汇总报告。
    核心输出：Alpha 捕获率——策略在随机下界与 Oracle 上界之间的位置。
    """
    def _safe(df: pd.DataFrame) -> Dict:
        if df.empty or "portfolio_ret" not in df.columns:
            return {}
        return _return_metrics(df["portfolio_ret"])

    strategy_m = _safe(strategy_daily)
    random_m   = _safe(random_daily)
    oracle_m   = _safe(oracle_daily)
    alpha_cap  = compute_alpha_capture(strategy_m, random_m, oracle_m)

    return {
        "description": (
            "Three-point coordinate system | "
            "random_baseline: zero model skill (lower bound) | "
            "strategy: actual result | "
            "oracle_ceiling: perfect prediction (upper bound)"
        ),
        "random_baseline": random_m,
        "strategy":        strategy_m,
        "oracle_ceiling":  oracle_m,
        "alpha_capture":   alpha_cap,
        "grade":           alpha_cap.get("overall_grade", "N/A"),
    }


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths",           default="configs/paths.yaml")
    ap.add_argument("--backtest_cfg",    default="configs/backtest.yaml")
    ap.add_argument("--run_id",          required=True)
    ap.add_argument("--split",           choices=["train", "val", "test"], default="test")
    ap.add_argument("--preds_path",      default=None)
    ap.add_argument("--model_artifacts", default=None)
    ap.add_argument("--skip_reference",  action="store_true",
                    help="跳过 random/oracle 参考模拟（调试加速用）")
    args = ap.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    bt_cfg    = load_yaml(Path(args.backtest_cfg))
    paths     = resolve_paths(paths_cfg, project_root=Path.cwd())

    run_dir    = paths.outputs_dir / args.run_id
    preds_path = _resolve_preds_path(run_dir, args.split, args.preds_path)
    split_path = paths.processed_dir / f"{args.split}.parquet"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    logger.info("loading predictions and split data")
    preds_df = _read_df(preds_path)
    split_df = _read_df(split_path)

    logger.info("building backtest panel")
    panel = _prepare_panel(preds_df, split_df, paths.tableB_path, int(bt_cfg["holding_td"]))
    execution_basis = infer_execution_basis(panel)
    ret_col = "exec_ret_1d"
    logger.info(f"panel rows={len(panel)}, execution_basis={execution_basis}")

    # stable 分组阈值
    threshold_ref_df = _load_named_splits(
        paths.processed_dir, threshold_reference_split_names(args.split)
    )
    stable_threshold_source = threshold_ref_df if not threshold_ref_df.empty else split_df
    stable_gap_cv_threshold = compute_stable_gap_cv_threshold(
        stable_threshold_source,
        stable_div_count_min=int(bt_cfg.get("stable_div_count_min", 4)),
        quantile=float(bt_cfg.get("stable_gap_cv_quantile", 0.5)),
    )
    logger.info(f"stable gap_cv threshold={stable_gap_cv_threshold:.6f}")

    # 当前 split 诊断（仅用于输出诊断报告，不影响规则决策）
    logger.info("running current-split research diagnostics")
    diagnostic_reports, diagnostic_decision = run_signal_research(
        panel, high_prob_threshold=float(bt_cfg.get("regular_prob_threshold", 0.55)),
    )

    # ── 分红规则激活决策 ──────────────────────────────────────────────────────
    mode            = str(bt_cfg.get("dividend_rules_mode", "auto")).lower()
    policy_decision = diagnostic_decision
    policy_source   = "current_split_diagnostic"

    if mode == "true":
        use_dividend_rules = True
        policy_source = "forced_true"
    elif mode == "false":
        use_dividend_rules = False
        policy_source = "forced_false"
    else:  # auto
        policy_ref_splits = policy_split_names(args.split)
        policy_ref_df     = _load_named_splits(paths.processed_dir, policy_ref_splits)
        if policy_ref_df.empty:
            use_dividend_rules = False
            policy_source = "no_heldout_policy_split_disable"
            logger.warning("auto mode: no held-out prior split; disabling dividend rules")
        else:
            # 优先复用已有预测文件，避免重新加载模型（解决 numpy 版本不兼容问题）
            val_preds_parquet = run_dir / "preds" / "val_preds.parquet"
            val_preds_csv     = run_dir / "preds" / "val_preds.csv"
            if val_preds_parquet.exists() or val_preds_csv.exists():
                val_preds_path = val_preds_parquet if val_preds_parquet.exists() else val_preds_csv
                logger.info(f"auto mode: using pre-computed val predictions: {val_preds_path}")
                val_preds_df = _read_df(val_preds_path)
                research_basis_panel = _prepare_panel(
                    val_preds_df, policy_ref_df, paths.tableB_path, int(bt_cfg["holding_td"])
                )
            else:
                # 备用：val 预测文件不存在时才重新加载模型
                art_path = _resolve_artifacts_path(paths.models_dir, args.run_id, args.model_artifacts)
                logger.info(f"auto mode: val preds not found, recomputing on: {policy_ref_splits}")
                research_basis_panel = _build_research_basis_panel(
                    policy_ref_df, art_path, paths.tableB_path, int(bt_cfg["holding_td"])
                )
            _, policy_decision = run_signal_research(
                research_basis_panel,
                high_prob_threshold=float(bt_cfg.get("regular_prob_threshold", 0.55)),
            )
            use_dividend_rules = bool(policy_decision.use_dividend_rules)
            policy_source = f"heldout_prior_splits:{','.join(policy_ref_splits)}"

    logger.info(
        f"dividend_rules_mode={mode}, active={use_dividend_rules}, "
        f"policy_source={policy_source}, "
        f"policy_votes={policy_decision.support_votes}/{policy_decision.total_checks}"
    )

    # ── 主策略 ────────────────────────────────────────────────────────────────
    logger.info("building strategy candidates")
    candidates = build_daily_candidates(
        panel=panel,
        top_k=int(bt_cfg["top_k"]),
        stable_gap_cv_threshold=stable_gap_cv_threshold,
        turnover_quantile_min=float(bt_cfg["turnover_quantile_min"]),
        exclude_div_count_le=int(bt_cfg["exclude_div_count_le"]),
        min_price=float(bt_cfg["min_price"]),
        stable_div_count_min=int(bt_cfg.get("stable_div_count_min", 4)),
        stable_prob_threshold=float(bt_cfg.get("stable_prob_threshold", 0.45)),
        regular_prob_threshold=float(bt_cfg.get("regular_prob_threshold", 0.55)),
        max_industry_weight=float(bt_cfg.get("max_industry_weight", 1.0)),
        use_dividend_rules=use_dividend_rules,
    )
    logger.info(f"strategy candidate rows={len(candidates)}")

    benchmark_df = build_equal_weight_benchmark(
        panel,
        min_price=float(bt_cfg["min_price"]),
        ret_col=ret_col,
        cost_bps_one_way=float(bt_cfg["cost_bps_one_way"]),
        holding_td=int(bt_cfg["holding_td"]),
        turnover_quantile_min=float(bt_cfg["turnover_quantile_min"]),
        exclude_div_count_le=int(bt_cfg["exclude_div_count_le"]),
    )

    logger.info("simulating portfolio")
    benchmark_df = build_equal_weight_benchmark(
        panel,
        min_price=float(bt_cfg["min_price"]),
        ret_col=ret_col,
        cost_bps_one_way=float(bt_cfg["cost_bps_one_way"]),           # Fix 1: 与策略成本口径对齐
        holding_td=int(bt_cfg["holding_td"]),                          # Fix 2: 与策略持仓期对齐
        turnover_quantile_min=float(bt_cfg["turnover_quantile_min"]),  # Fix 3: 与策略宇宙对齐
        exclude_div_count_le=int(bt_cfg["exclude_div_count_le"]),      # Fix 3: 与策略宇宙对齐
    )
    daily_df, trades_df, positions_df = simulate_portfolio(
        panel=panel,
        candidates=candidates,
        top_k=int(bt_cfg["top_k"]),
        holding_td=int(bt_cfg["holding_td"]),
        cooldown_td=int(bt_cfg["cooldown_td"]),
        cost_bps_one_way=float(bt_cfg["cost_bps_one_way"]),
        max_industry_weight=float(bt_cfg.get("max_industry_weight", 1.0)),
        ret_col=ret_col,
        use_bid_ask_spread=bool(bt_cfg.get("use_bid_ask_spread", False)),
        spread_cost_cap_bps_one_way=float(bt_cfg.get("spread_cost_cap_bps_one_way", 100.0)),
    )
    daily_df = enrich_daily_report(daily_df, benchmark_df)
    logger.info(f"strategy: daily rows={len(daily_df)}, trades={len(trades_df)}")

    # ── 三点坐标系：随机基准 + Oracle 基准 ────────────────────────────────────
    three_way     = {}
    random_daily  = pd.DataFrame()
    oracle_daily  = pd.DataFrame()
    random_trades = pd.DataFrame()
    oracle_trades = pd.DataFrame()

    if not args.skip_reference:
        common_kw = dict(
            min_price=float(bt_cfg["min_price"]),
            turnover_quantile_min=float(bt_cfg["turnover_quantile_min"]),
            exclude_div_count_le=int(bt_cfg["exclude_div_count_le"]),
        )

        random_candidates = build_random_candidates(
            panel=panel,
            top_k=int(bt_cfg["top_k"]),
            seed=int(bt_cfg.get("random_seed", 42)),
            **common_kw,
        )
        random_daily, random_trades = _simulate_reference(
            "random_baseline", random_candidates, panel, bt_cfg, ret_col
        )

        oracle_candidates = build_oracle_candidates(
            panel=panel,
            top_k=int(bt_cfg["top_k"]),
            **common_kw,
        )
        oracle_daily, oracle_trades = _simulate_reference(
            "oracle_ceiling", oracle_candidates, panel, bt_cfg, ret_col
        )

        if not random_daily.empty and not oracle_daily.empty:
            three_way = _build_three_way_comparison(daily_df, random_daily, oracle_daily)
            logger.info(
                "Alpha capture ratio: "
                f"{three_way.get('alpha_capture', {}).get('primary_alpha_capture_ratio', 'N/A')}"
            )
            logger.info(f"Grade: {three_way.get('grade', 'N/A')}")
        else:
            logger.warning("random or oracle simulation empty; skipping three-way comparison")
    else:
        logger.info("--skip_reference: skipping random/oracle simulations")

    # ── 汇总与输出 ─────────────────────────────────────────────────────────────
    summary = summarize_backtest(
        daily_df,
        trades_df,
        _decision_payload(policy_decision, diagnostic_decision, mode,
                          use_dividend_rules, policy_source, execution_basis),
        cost_model=_cost_model_payload(bt_cfg, execution_basis),
        three_way_comparison=three_way,
    )
    attribution = build_trade_attribution(trades_df)

    out_dir = run_dir / "backtest"
    write_backtest_outputs(
        out_dir=out_dir,
        panel=panel,
        candidates=candidates,
        daily_df=daily_df,
        trades_df=trades_df,
        positions_df=positions_df,
        summary=summary,
        research_reports=diagnostic_reports,
        attribution_reports=attribution,
        random_daily_df=random_daily,
        oracle_daily_df=oracle_daily,
        random_trades_df=random_trades,
        oracle_trades_df=oracle_trades,
    )
    logger.info(f"wrote backtest outputs under: {out_dir}")


if __name__ == "__main__":
    main()