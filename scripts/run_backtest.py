from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.append(str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

from src.backtest.benchmark import build_equal_weight_benchmark
from src.backtest.portfolio import simulate_portfolio
from src.backtest.report import build_trade_attribution, enrich_daily_report, summarize_backtest, write_backtest_outputs
from src.backtest.signal import add_forward_returns, build_backtest_panel, build_daily_candidates, compute_stable_gap_cv_threshold, run_signal_research
from src.utils.logging import get_logger
from src.utils.paths import load_yaml, resolve_paths


logger = get_logger("run_backtest")


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


def _reference_df(processed_dir: Path, split: str) -> pd.DataFrame:
    parts = []
    for name in ("train", "val"):
        if name == split:
            continue
        path = processed_dir / f"{name}.parquet"
        if path.exists():
            parts.append(_read_df(path))
    if not parts:
        return _read_df(processed_dir / f"{split}.parquet")
    return pd.concat(parts, ignore_index=True)


def _decision_payload(decision, mode: str) -> Dict:
    return {
        "mode": mode,
        "use_dividend_rules": bool(decision.use_dividend_rules),
        "support_votes": int(decision.support_votes),
        "total_checks": int(decision.total_checks),
        "checks": decision.checks,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--backtest_cfg", default="configs/backtest.yaml")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--preds_path", default=None)
    args = ap.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    bt_cfg = load_yaml(Path(args.backtest_cfg))
    paths = resolve_paths(paths_cfg, project_root=Path.cwd())

    run_dir = paths.outputs_dir / args.run_id
    preds_path = _resolve_preds_path(run_dir, args.split, args.preds_path)
    split_path = paths.processed_dir / f"{args.split}.parquet"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    preds_df = _read_df(preds_path)
    split_df = _read_df(split_path)
    ref_df = _reference_df(paths.processed_dir, args.split)

    panel = build_backtest_panel(preds_df, split_df)
    panel = add_forward_returns(panel, horizons=(1, 5, int(bt_cfg["holding_td"])))

    stable_gap_cv_threshold = compute_stable_gap_cv_threshold(
        ref_df,
        stable_div_count_min=int(bt_cfg.get("stable_div_count_min", 4)),
        quantile=float(bt_cfg.get("stable_gap_cv_quantile", 0.5)),
    )
    logger.info(f"stable gap_cv threshold={stable_gap_cv_threshold:.6f}")

    research_reports, decision = run_signal_research(panel, high_prob_threshold=float(bt_cfg.get("regular_prob_threshold", 0.55)))

    mode = str(bt_cfg.get("dividend_rules_mode", "auto")).lower()
    if mode == "true":
        use_dividend_rules = True
    elif mode == "false":
        use_dividend_rules = False
    else:
        use_dividend_rules = bool(decision.use_dividend_rules)

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

    benchmark_df = build_equal_weight_benchmark(panel, min_price=float(bt_cfg["min_price"]))
    daily_df, trades_df, positions_df = simulate_portfolio(
        panel=panel,
        candidates=candidates,
        top_k=int(bt_cfg["top_k"]),
        holding_td=int(bt_cfg["holding_td"]),
        cooldown_td=int(bt_cfg["cooldown_td"]),
        cost_bps_one_way=float(bt_cfg["cost_bps_one_way"]),
        max_industry_weight=float(bt_cfg.get("max_industry_weight", 1.0)),
    )
    daily_df = enrich_daily_report(daily_df, benchmark_df)

    summary = summarize_backtest(daily_df, trades_df, _decision_payload(decision, mode))
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
        research_reports=research_reports,
        attribution_reports=attribution,
    )
    logger.info(f"wrote backtest outputs under: {out_dir}")


if __name__ == "__main__":
    main()
