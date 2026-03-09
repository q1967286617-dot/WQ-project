from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.append(str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.backtest.benchmark import build_equal_weight_benchmark
from src.backtest.portfolio import simulate_portfolio
from src.backtest.report import build_trade_attribution, enrich_daily_report, summarize_backtest, write_backtest_outputs
from src.backtest.signal import add_execution_returns, add_forward_returns, build_backtest_panel, build_daily_candidates, compute_stable_gap_cv_threshold, infer_execution_basis, merge_execution_price_data, run_signal_research
from src.modeling.predict import predict_to_eval_df
from src.utils.logging import get_logger
from src.utils.paths import load_yaml, resolve_paths


logger = get_logger("run_backtest")


KEEP_EXTRA_COLS = ["log_mkt_cap", "turnover_5d", "vol_21d", "SICCD", "industry", "has_div_history"]
RAW_PRICE_COLS = ["PERMNO", "DlyCalDt", "DlyOpen", "DlyClose", "DlyHigh", "DlyLow", "DlyBid", "DlyAsk"]


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
    # Use only held-out, prior-in-time validation-style splits for rule selection.
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
    parquet = table_b_path.with_suffix('.parquet')
    return parquet if parquet.exists() else table_b_path


def _load_execution_price_data(table_b_path: Path, base_df: pd.DataFrame) -> pd.DataFrame:
    if base_df.empty:
        return pd.DataFrame(columns=["date", "permno", "DlyOpen", "DlyClose", "DlyHigh", "DlyLow", "DlyBid", "DlyAsk"])

    source = _tableb_source_path(table_b_path)
    permnos = set(pd.to_numeric(base_df["PERMNO"], errors="coerce").dropna().astype(int).unique().tolist())
    min_date = pd.to_datetime(base_df["DlyCalDt"], errors="coerce").min()
    max_date = pd.to_datetime(base_df["DlyCalDt"], errors="coerce").max()

    if source.suffix.lower() == ".parquet":
        raw = pd.read_parquet(source, columns=RAW_PRICE_COLS)
        raw["DlyCalDt"] = pd.to_datetime(raw["DlyCalDt"], errors="coerce")
        raw["PERMNO"] = pd.to_numeric(raw["PERMNO"], errors="coerce").astype("Int64")
        raw = raw[raw["PERMNO"].isin(permnos) & raw["DlyCalDt"].between(min_date, max_date)].copy()
    else:
        parts = []
        reader = pd.read_csv(source, usecols=RAW_PRICE_COLS, chunksize=250000)
        for chunk in reader:
            chunk["DlyCalDt"] = pd.to_datetime(chunk["DlyCalDt"], errors="coerce")
            chunk["PERMNO"] = pd.to_numeric(chunk["PERMNO"], errors="coerce").astype("Int64")
            chunk = chunk[chunk["PERMNO"].isin(permnos) & chunk["DlyCalDt"].between(min_date, max_date)]
            if not chunk.empty:
                parts.append(chunk)
        raw = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=RAW_PRICE_COLS)

    raw = raw.rename(columns={"PERMNO": "permno", "DlyCalDt": "date"})
    raw["permno"] = pd.to_numeric(raw["permno"], errors="coerce").astype(int)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    return raw.drop_duplicates(subset=["date", "permno"]).sort_values(["date", "permno"]).reset_index(drop=True)


def _prepare_panel(base_preds: pd.DataFrame, base_split: pd.DataFrame, table_b_path: Path, holding_td: int) -> pd.DataFrame:
    panel = build_backtest_panel(base_preds, base_split)
    price_df = _load_execution_price_data(table_b_path, base_split)
    panel = merge_execution_price_data(panel, price_df)
    panel = add_execution_returns(panel)
    panel = add_forward_returns(panel, horizons=(1, 5, int(holding_td)))
    return panel


def _build_research_basis_panel(ref_df: pd.DataFrame, art_path: Path, table_b_path: Path, holding_td: int) -> pd.DataFrame:
    ref_preds = predict_to_eval_df(ref_df, art_path, keep_extra_cols=KEEP_EXTRA_COLS)
    return _prepare_panel(ref_preds, ref_df, table_b_path, holding_td)


def _decision_payload(policy_decision, diagnostic_decision, mode: str, active_value: bool, policy_source: str, execution_basis: str) -> Dict:
    return {
        "mode": mode,
        "use_dividend_rules": bool(active_value),
        "policy_source": policy_source,
        "execution_basis": execution_basis,
        "research_recommended_dividend_rules": bool(policy_decision.use_dividend_rules),
        "support_votes": int(policy_decision.support_votes),
        "total_checks": int(policy_decision.total_checks),
        "checks": policy_decision.checks,
        "current_split_diagnostic": {
            "recommended_dividend_rules": bool(diagnostic_decision.use_dividend_rules),
            "support_votes": int(diagnostic_decision.support_votes),
            "total_checks": int(diagnostic_decision.total_checks),
            "checks": diagnostic_decision.checks,
        },
    }

def _cost_model_payload(bt_cfg: Dict, execution_basis: str) -> Dict:
    use_spread = bool(bt_cfg.get("use_bid_ask_spread", False))
    spread_cap = float(bt_cfg.get("spread_cost_cap_bps_one_way", 100.0)) if use_spread else None
    return {
        "execution_basis": execution_basis,
        "fixed_cost_bps_one_way": float(bt_cfg.get("cost_bps_one_way", 0.0)),
        "use_bid_ask_spread": use_spread,
        "spread_cost_model": "half_spread_from_daily_bid_ask" if use_spread else "disabled",
        "spread_cost_cap_bps_one_way": spread_cap,
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--backtest_cfg", default="configs/backtest.yaml")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--preds_path", default=None)
    ap.add_argument("--model_artifacts", default=None)
    args = ap.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    bt_cfg = load_yaml(Path(args.backtest_cfg))
    paths = resolve_paths(paths_cfg, project_root=Path.cwd())

    run_dir = paths.outputs_dir / args.run_id
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

    threshold_ref_df = _load_named_splits(paths.processed_dir, threshold_reference_split_names(args.split))
    stable_threshold_source = threshold_ref_df if not threshold_ref_df.empty else split_df
    stable_gap_cv_threshold = compute_stable_gap_cv_threshold(
        stable_threshold_source,
        stable_div_count_min=int(bt_cfg.get("stable_div_count_min", 4)),
        quantile=float(bt_cfg.get("stable_gap_cv_quantile", 0.5)),
    )
    logger.info(f"stable gap_cv threshold={stable_gap_cv_threshold:.6f}")

    logger.info("running current-split research diagnostics")
    diagnostic_reports, diagnostic_decision = run_signal_research(
        panel,
        high_prob_threshold=float(bt_cfg.get("regular_prob_threshold", 0.55)),
    )

    mode = str(bt_cfg.get("dividend_rules_mode", "auto")).lower()
    policy_decision = diagnostic_decision
    policy_source = "current_split_diagnostic"

    if mode == "true":
        use_dividend_rules = True
        policy_source = "forced_true"
    elif mode == "false":
        use_dividend_rules = False
        policy_source = "forced_false"
    else:
        policy_ref_splits = policy_split_names(args.split)
        policy_ref_df = _load_named_splits(paths.processed_dir, policy_ref_splits)
        if policy_ref_df.empty:
            use_dividend_rules = False
            policy_source = "no_heldout_policy_split_disable"
            logger.warning("auto mode has no held-out prior split available; disabling dividend rules to avoid in-sample policy selection")
        else:
            art_path = _resolve_artifacts_path(paths.models_dir, args.run_id, args.model_artifacts)
            logger.info(f"running policy selection research on held-out prior splits: {policy_ref_splits}")
            research_basis_panel = _build_research_basis_panel(policy_ref_df, art_path, paths.tableB_path, int(bt_cfg["holding_td"]))
            _, policy_decision = run_signal_research(
                research_basis_panel,
                high_prob_threshold=float(bt_cfg.get("regular_prob_threshold", 0.55)),
            )
            use_dividend_rules = bool(policy_decision.use_dividend_rules)
            policy_source = f"heldout_prior_splits:{','.join(policy_ref_splits)}"
    logger.info(
        f"dividend_rules_mode={mode}, active={use_dividend_rules}, policy_source={policy_source}, policy_votes={policy_decision.support_votes}/{policy_decision.total_checks}"
    )

    logger.info("building daily candidates")
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
    logger.info(f"candidate rows={len(candidates)}")

    logger.info("simulating portfolio")
    benchmark_df = build_equal_weight_benchmark(panel, min_price=float(bt_cfg["min_price"]), ret_col=ret_col)
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
    logger.info(f"daily rows={len(daily_df)}, trades={len(trades_df)}, positions={len(positions_df)}")

    summary = summarize_backtest(
        daily_df,
        trades_df,
        _decision_payload(policy_decision, diagnostic_decision, mode, use_dividend_rules, policy_source, execution_basis),
        cost_model=_cost_model_payload(bt_cfg, execution_basis),
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
    )
    logger.info(f"wrote backtest outputs under: {out_dir}")


if __name__ == "__main__":
    main()


