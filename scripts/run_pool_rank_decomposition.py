from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.backtest.benchmark import build_equal_weight_benchmark  # noqa: E402
from src.backtest.portfolio import simulate_portfolio  # noqa: E402
from src.backtest.report import enrich_daily_report, summarize_backtest  # noqa: E402
from src.backtest.signal import (  # noqa: E402
    attach_score_column,
    build_backtest_panel,
    compute_stable_gap_cv_threshold,
    infer_execution_basis,
    merge_execution_price_data,
    prepare_candidate_pool,
    run_signal_research,
    select_top_k_from_pool_by_score,
)
from src.eval.report import dump_df, dump_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import ensure_dir, load_yaml, resolve_paths  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
logger = get_logger("run_pool_rank_decomposition")


def _read_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            csv_path = path.with_suffix(".csv")
            if csv_path.exists():
                return pd.read_csv(csv_path)
            raise RuntimeError(
                f"Failed to read parquet: {path}. Install pyarrow (recommended) or convert to CSV.\n"
                f"Original error: {exc}"
            )
    return pd.read_csv(path)


def _resolve_preds_path(run_dir: Path | None, split: str, override: str | None) -> Path:
    if override:
        return Path(override)
    if run_dir is None:
        raise FileNotFoundError("Missing run_dir for prediction lookup")
    parquet = run_dir / "preds" / f"{split}_preds.parquet"
    csv = parquet.with_suffix(".csv")
    if parquet.exists():
        return parquet
    if csv.exists():
        return csv
    raise FileNotFoundError(f"Missing predictions for split={split}: {parquet}")


def _infer_run_dir(run_id: str | None, preds_path: str | None, outputs_dir: Path) -> Path | None:
    if run_id:
        return outputs_dir / run_id
    if preds_path:
        p = Path(preds_path).resolve()
        if p.parent.name == "preds" and p.parent.parent.name:
            return p.parent.parent
    return None


def _load_named_splits(processed_dir: Path, split_names: list[str]) -> pd.DataFrame:
    parts = []
    for name in split_names:
        parquet_path = processed_dir / f"{name}.parquet"
        csv_path = processed_dir / f"{name}.csv"
        if parquet_path.exists():
            parts.append(_read_df(parquet_path))
        elif csv_path.exists():
            parts.append(_read_df(csv_path))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _policy_split_names(target_split: str) -> list[str]:
    if target_split == "test":
        return ["val"]
    return []


def _threshold_reference_split_names(target_split: str) -> list[str]:
    order = ["train", "val", "test"]
    idx = order.index(target_split)
    return order[:idx]


def _tableb_source_path(table_b_path: Path) -> Path:
    parquet = table_b_path.with_suffix(".parquet")
    return parquet if parquet.exists() else table_b_path


def _load_execution_price_data(table_b_path: Path, base_df: pd.DataFrame) -> pd.DataFrame:
    raw_cols = ["PERMNO", "DlyCalDt", "DlyOpen", "DlyClose", "DlyHigh", "DlyLow", "DlyBid", "DlyAsk"]
    if base_df.empty:
        return pd.DataFrame(columns=["date", "permno", "DlyOpen", "DlyClose", "DlyHigh", "DlyLow", "DlyBid", "DlyAsk"])

    source = _tableb_source_path(table_b_path)
    permnos = set(pd.to_numeric(base_df["PERMNO"], errors="coerce").dropna().astype(int).unique().tolist())
    min_date = pd.to_datetime(base_df["DlyCalDt"], errors="coerce").min()
    max_date = pd.to_datetime(base_df["DlyCalDt"], errors="coerce").max()

    if source.suffix.lower() == ".parquet":
        raw = pd.read_parquet(source, columns=raw_cols)
        raw["DlyCalDt"] = pd.to_datetime(raw["DlyCalDt"], errors="coerce")
        raw["PERMNO"] = pd.to_numeric(raw["PERMNO"], errors="coerce").astype("Int64")
        raw = raw[raw["PERMNO"].isin(permnos) & raw["DlyCalDt"].between(min_date, max_date)].copy()
    else:
        parts = []
        reader = pd.read_csv(source, usecols=raw_cols, chunksize=250000)
        for chunk in reader:
            chunk["DlyCalDt"] = pd.to_datetime(chunk["DlyCalDt"], errors="coerce")
            chunk["PERMNO"] = pd.to_numeric(chunk["PERMNO"], errors="coerce").astype("Int64")
            chunk = chunk[chunk["PERMNO"].isin(permnos) & chunk["DlyCalDt"].between(min_date, max_date)]
            if not chunk.empty:
                parts.append(chunk)
        raw = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=raw_cols)

    raw = raw.rename(columns={"PERMNO": "permno", "DlyCalDt": "date"})
    raw["permno"] = pd.to_numeric(raw["permno"], errors="coerce").astype(int)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    return raw.drop_duplicates(subset=["date", "permno"]).sort_values(["date", "permno"]).reset_index(drop=True)


def _prepare_panel(preds_df: pd.DataFrame, split_df: pd.DataFrame, table_b_path: Path, holding_td: int) -> pd.DataFrame:
    panel = build_backtest_panel(preds_df, split_df)
    price_df = _load_execution_price_data(table_b_path, split_df)
    panel = merge_execution_price_data(panel, price_df)
    panel = add_execution_returns(panel)
    horizons = sorted({1, 5, 10, int(holding_td)})
    panel = add_forward_returns(panel, horizons=horizons)
    return panel


def add_execution_returns(panel: pd.DataFrame) -> pd.DataFrame:
    from src.backtest.signal import add_execution_returns as _add_execution_returns

    return _add_execution_returns(panel)


def add_forward_returns(panel: pd.DataFrame, horizons: tuple[int, ...] | list[int]) -> pd.DataFrame:
    from src.backtest.signal import add_forward_returns as _add_forward_returns

    return _add_forward_returns(panel, horizons=horizons)


def _resolve_use_dividend_rules(
    pool_panel: pd.DataFrame,
    backtest_cfg: dict[str, Any],
    target_split: str,
    processed_dir: Path,
    source_run_dir: Path | None,
) -> tuple[bool, str, Dict[str, Any]]:
    mode = str(backtest_cfg.get("dividend_rules_mode", "auto")).lower()
    if mode == "true":
        return True, "forced_true", {}
    if mode == "false":
        return False, "forced_false", {}

    policy_ref_splits = _policy_split_names(target_split)
    policy_ref_df = _load_named_splits(processed_dir, policy_ref_splits)
    if not policy_ref_splits or policy_ref_df.empty:
        reports, decision = run_signal_research(
            pool_panel,
            high_prob_threshold=float(backtest_cfg.get("regular_prob_threshold", 0.55)),
        )
        return bool(decision.use_dividend_rules), "current_split_diagnostic", {
            "reports": reports,
            "decision": decision,
        }

    if source_run_dir is not None:
        for ref_split in policy_ref_splits:
            candidate_path = source_run_dir / "preds" / f"{ref_split}_preds.parquet"
            if not candidate_path.exists():
                candidate_path = candidate_path.with_suffix(".csv")
            if candidate_path.exists():
                ref_preds = _read_df(candidate_path)
                ref_panel = _prepare_panel(ref_preds, policy_ref_df, backtest_cfg["_table_b_path"], int(backtest_cfg["holding_td"]))
                reports, decision = run_signal_research(
                    ref_panel,
                    high_prob_threshold=float(backtest_cfg.get("regular_prob_threshold", 0.55)),
                )
                return bool(decision.use_dividend_rules), f"heldout_prior_splits:{','.join(policy_ref_splits)}", {
                    "reports": reports,
                    "decision": decision,
                }

    reports, decision = run_signal_research(
        pool_panel,
        high_prob_threshold=float(backtest_cfg.get("regular_prob_threshold", 0.55)),
    )
    return bool(decision.use_dividend_rules), "current_split_diagnostic", {"reports": reports, "decision": decision}


def _build_combo(
    *,
    combo_name: str,
    pool_label: str,
    rank_label: str,
    pool_panel: pd.DataFrame,
    rank_preds_df: pd.DataFrame,
    paths,
    backtest_cfg: dict[str, Any],
    stable_gap_cv_threshold: float,
    use_dividend_rules: bool,
    top_k: int,
) -> dict[str, Any]:
    pool = prepare_candidate_pool(
        panel=pool_panel,
        stable_gap_cv_threshold=stable_gap_cv_threshold,
        turnover_quantile_min=float(backtest_cfg["turnover_quantile_min"]),
        exclude_div_count_le=int(backtest_cfg["exclude_div_count_le"]),
        min_price=float(backtest_cfg["min_price"]),
        stable_div_count_min=int(backtest_cfg.get("stable_div_count_min", 4)),
        stable_prob_threshold=float(backtest_cfg.get("stable_prob_threshold", 0.45)),
        regular_prob_threshold=float(backtest_cfg.get("regular_prob_threshold", 0.55)),
        use_dividend_rules=use_dividend_rules,
    )
    if pool.empty:
        return {"combo": combo_name, "pool_label": pool_label, "rank_label": rank_label, "summary": {}, "candidate_rows": 0}

    pool = pool.copy()
    pool["pool_prob"] = pool["prob"]
    pool = attach_score_column(pool, rank_preds_df, score_col="prob", output_col="rank_prob", strict=True)
    selected = select_top_k_from_pool_by_score(
        pool=pool,
        top_k=top_k,
        max_industry_weight=float(backtest_cfg.get("max_industry_weight", 1.0)),
        score_col="rank_prob",
    )
    if selected.empty:
        return {"combo": combo_name, "pool_label": pool_label, "rank_label": rank_label, "summary": {}, "candidate_rows": 0}

    selected = selected.copy()
    selected["prob"] = selected["rank_prob"]

    benchmark_df = build_equal_weight_benchmark(
        pool_panel,
        min_price=float(backtest_cfg["min_price"]),
        ret_col="exec_ret_1d",
        cost_bps_one_way=float(backtest_cfg["cost_bps_one_way"]),
        holding_td=int(backtest_cfg["holding_td"]),
        turnover_quantile_min=float(backtest_cfg["turnover_quantile_min"]),
        exclude_div_count_le=int(backtest_cfg["exclude_div_count_le"]),
    )

    daily_df, trades_df, positions_df = simulate_portfolio(
        panel=pool_panel,
        candidates=selected,
        top_k=top_k,
        holding_td=int(backtest_cfg["holding_td"]),
        cooldown_td=int(backtest_cfg["cooldown_td"]),
        cost_bps_one_way=float(backtest_cfg["cost_bps_one_way"]),
        max_industry_weight=float(backtest_cfg.get("max_industry_weight", 1.0)),
        ret_col="exec_ret_1d",
        use_bid_ask_spread=bool(backtest_cfg.get("use_bid_ask_spread", False)),
        spread_cost_cap_bps_one_way=float(backtest_cfg.get("spread_cost_cap_bps_one_way", 100.0)),
    )
    daily_df = enrich_daily_report(daily_df, benchmark_df)

    research_decision = {
        "mode": "pool_rank_decomposition",
        "use_dividend_rules": bool(use_dividend_rules),
        "pool_source": pool_label,
        "rank_source": rank_label,
        "candidate_rows": int(len(selected)),
    }
    summary = summarize_backtest(
        daily_df,
        trades_df,
        research_decision,
        cost_model={
            "execution_basis": infer_execution_basis(pool_panel),
            "fixed_cost_bps_one_way": float(backtest_cfg["cost_bps_one_way"]),
            "use_bid_ask_spread": bool(backtest_cfg.get("use_bid_ask_spread", False)),
            "spread_cost_model": "half_spread_from_daily_bid_ask" if bool(backtest_cfg.get("use_bid_ask_spread", False)) else "disabled",
            "spread_cost_cap_bps_one_way": float(backtest_cfg.get("spread_cost_cap_bps_one_way", 100.0)) if bool(backtest_cfg.get("use_bid_ask_spread", False)) else None,
        },
    )
    summary["candidate_rows"] = int(len(selected))
    summary["pool_rows"] = int(len(pool))

    out_dir = PROJECT_ROOT / "outputs" / "pool_rank_decomposition_outputs" / combo_name
    ensure_dir(out_dir)
    dump_df(pool, out_dir / "pool_candidates.csv")
    dump_df(selected, out_dir / "signals.csv")
    dump_df(daily_df, out_dir / "daily_portfolio.csv")
    dump_df(trades_df, out_dir / "trades.csv")
    dump_df(positions_df, out_dir / "positions.csv")
    dump_json(summary, out_dir / "summary.json")

    return {
        "combo": combo_name,
        "pool_label": pool_label,
        "rank_label": rank_label,
        "summary": summary,
        "candidate_rows": int(len(selected)),
        "pool_rows": int(len(pool)),
        "out_dir": str(out_dir),
    }


def _metric_slice(summary: Dict[str, Any]) -> Dict[str, float]:
    portfolio = summary.get("portfolio", {}) if summary else {}
    return {
        "annualized_return": float(portfolio.get("annualized_return", 0.0)),
        "sharpe": float(portfolio.get("sharpe", 0.0)),
        "total_return": float(portfolio.get("total_return", 0.0)),
        "max_drawdown": float(portfolio.get("max_drawdown", 0.0)),
    }


def _delta(lhs: Dict[str, float], rhs: Dict[str, float]) -> Dict[str, float]:
    return {f"{k}_diff": float(lhs[k]) - float(rhs[k]) for k in lhs.keys()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pool-vs-ranking decomposition for two prediction sources.")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--backtest_cfg", default="configs/backtest.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--run_id_a", default=None)
    parser.add_argument("--run_id_b", default=None)
    parser.add_argument("--preds_a", default=None)
    parser.add_argument("--preds_b", default=None)
    parser.add_argument("--analysis_id", default=None)
    parser.add_argument("--allow_partial_overlap", action="store_true")
    args = parser.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    backtest_cfg = load_yaml(Path(args.backtest_cfg))
    paths = resolve_paths(paths_cfg, project_root=PROJECT_ROOT)
    backtest_cfg["_table_b_path"] = paths.tableB_path

    run_dir_a = _infer_run_dir(args.run_id_a, args.preds_a, paths.outputs_dir)
    run_dir_b = _infer_run_dir(args.run_id_b, args.preds_b, paths.outputs_dir)
    preds_path_a = _resolve_preds_path(run_dir_a, args.split, args.preds_a)
    preds_path_b = _resolve_preds_path(run_dir_b, args.split, args.preds_b)

    split_path = paths.processed_dir / f"{args.split}.parquet"
    if not split_path.exists():
        csv_split_path = paths.processed_dir / f"{args.split}.csv"
        if csv_split_path.exists():
            split_path = csv_split_path
        else:
            raise FileNotFoundError(f"Missing split file: {split_path}")

    split_df = _read_df(split_path)
    preds_a = _read_df(preds_path_a)
    preds_b = _read_df(preds_path_b)

    panel_a = _prepare_panel(preds_a, split_df, paths.tableB_path, int(backtest_cfg["holding_td"]))
    if args.allow_partial_overlap:
        panel_b = _prepare_panel(preds_b, split_df, paths.tableB_path, int(backtest_cfg["holding_td"]))
        common_keys = (
            panel_a[["date", "permno"]]
            .drop_duplicates()
            .merge(panel_b[["date", "permno"]].drop_duplicates(), on=["date", "permno"], how="inner")
        )
        panel_a = panel_a.merge(common_keys, on=["date", "permno"], how="inner")
        panel_b = panel_b.merge(common_keys, on=["date", "permno"], how="inner")
    else:
        panel_b = _prepare_panel(preds_b, split_df, paths.tableB_path, int(backtest_cfg["holding_td"]))
        keys_a = panel_a[["date", "permno"]].drop_duplicates().sort_values(["date", "permno"]).reset_index(drop=True)
        keys_b = panel_b[["date", "permno"]].drop_duplicates().sort_values(["date", "permno"]).reset_index(drop=True)
        if not keys_a.equals(keys_b):
            diff_a = int(len(keys_a.merge(keys_b, on=["date", "permno"], how="left", indicator=True).query("_merge == 'left_only'")))
            diff_b = int(len(keys_b.merge(keys_a, on=["date", "permno"], how="left", indicator=True).query("_merge == 'left_only'")))
            raise ValueError(
                "A/B panels do not have identical keys. "
                f"Missing in B: {diff_a}, missing in A: {diff_b}. "
                "Pass --allow_partial_overlap if you want to analyze the intersection only."
            )

    stable_threshold_source = _load_named_splits(paths.processed_dir, _threshold_reference_split_names(args.split))
    if stable_threshold_source.empty:
        stable_threshold_source = split_df
    stable_gap_cv_threshold = compute_stable_gap_cv_threshold(
        stable_threshold_source,
        stable_div_count_min=int(backtest_cfg.get("stable_div_count_min", 4)),
        quantile=float(backtest_cfg.get("stable_gap_cv_quantile", 0.5)),
    )

    use_rules_a, policy_source_a, policy_meta_a = _resolve_use_dividend_rules(
        panel_a,
        backtest_cfg,
        args.split,
        paths.processed_dir,
        run_dir_a,
    )
    use_rules_b, policy_source_b, policy_meta_b = _resolve_use_dividend_rules(
        panel_b,
        backtest_cfg,
        args.split,
        paths.processed_dir,
        run_dir_b,
    )

    top_k = int(backtest_cfg["top_k"])
    combos = [
        ("pool_a_rank_a", "A", "A", panel_a, preds_a, bool(use_rules_a)),
        ("pool_a_rank_b", "A", "B", panel_a, preds_b, bool(use_rules_a)),
        ("pool_b_rank_a", "B", "A", panel_b, preds_a, bool(use_rules_b)),
        ("pool_b_rank_b", "B", "B", panel_b, preds_b, bool(use_rules_b)),
    ]

    results: list[dict[str, Any]] = []
    for combo_name, pool_label, rank_label, pool_panel, rank_preds, use_rules in combos:
        logger.info(f"running {combo_name}")
        results.append(
            _build_combo(
                combo_name=combo_name,
                pool_label=pool_label,
                rank_label=rank_label,
                pool_panel=pool_panel,
                rank_preds_df=rank_preds,
                paths=paths,
                backtest_cfg=backtest_cfg,
                stable_gap_cv_threshold=stable_gap_cv_threshold,
                use_dividend_rules=use_rules,
                top_k=top_k,
            )
        )

    metric_map = {
        item["combo"]: _metric_slice(item.get("summary", {}))
        for item in results
    }
    base = metric_map["pool_a_rank_a"]
    decomposition = {
        "pool_effect_b_minus_a_given_rank_a": _delta(metric_map["pool_b_rank_a"], metric_map["pool_a_rank_a"]),
        "rank_effect_b_minus_a_given_pool_a": _delta(metric_map["pool_a_rank_b"], metric_map["pool_a_rank_a"]),
        "interaction": _delta(metric_map["pool_b_rank_b"], metric_map["pool_b_rank_a"]),
    }
    decomposition["interaction"] = {
        k: float(metric_map["pool_b_rank_b"][k] - metric_map["pool_b_rank_a"][k] - metric_map["pool_a_rank_b"][k] + base[k])
        for k in base.keys()
    }

    analysis_id = args.analysis_id or f"{Path(preds_path_a).stem}_vs_{Path(preds_path_b).stem}_{args.split}"
    out_dir = PROJECT_ROOT / "outputs" / "pool_rank_decomposition_outputs" / analysis_id
    ensure_dir(out_dir)

    summary = {
        "analysis_id": analysis_id,
        "split": args.split,
        "sources": {
            "A": {
                "run_id": args.run_id_a,
                "preds_path": str(preds_path_a),
                "policy_source": policy_source_a,
                "use_dividend_rules": bool(use_rules_a),
                "policy_meta_present": bool(policy_meta_a),
            },
            "B": {
                "run_id": args.run_id_b,
                "preds_path": str(preds_path_b),
                "policy_source": policy_source_b,
                "use_dividend_rules": bool(use_rules_b),
                "policy_meta_present": bool(policy_meta_b),
            },
        },
        "shared_config": {
            "top_k": top_k,
            "holding_td": int(backtest_cfg["holding_td"]),
            "cooldown_td": int(backtest_cfg["cooldown_td"]),
            "turnover_quantile_min": float(backtest_cfg["turnover_quantile_min"]),
            "exclude_div_count_le": int(backtest_cfg["exclude_div_count_le"]),
            "min_price": float(backtest_cfg["min_price"]),
            "max_industry_weight": float(backtest_cfg.get("max_industry_weight", 1.0)),
            "use_bid_ask_spread": bool(backtest_cfg.get("use_bid_ask_spread", False)),
        },
        "combos": {item["combo"]: item["summary"] for item in results},
        "combo_meta": {item["combo"]: {"pool_label": item["pool_label"], "rank_label": item["rank_label"], "candidate_rows": item["candidate_rows"], "pool_rows": item["pool_rows"], "out_dir": item["out_dir"]} for item in results},
        "metric_slice": metric_map,
        "decomposition": decomposition,
    }

    dump_json(summary, out_dir / "summary.json")
    dump_df(pd.DataFrame([{"combo": k, **v} for k, v in metric_map.items()]), out_dir / "metrics_table.csv")
    logger.info(f"wrote analysis outputs to: {out_dir}")


if __name__ == "__main__":
    main()
