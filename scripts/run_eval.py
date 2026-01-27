from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.utils.paths import load_yaml, resolve_paths, ensure_dir
from src.utils.logging import get_logger
from src.data.load import load_div_events, standardize_events
from src.eval.eval_tools import (
    validate_eval_df,
    global_metrics,
    stock_aucpr_best_worst,
    daily_topk_report,
    make_daily_topk_alerts,
    filter_events_for_eval,
    event_level_report_v2,
    evaluate_alerts_forward_window,
    compute_cadence_stats,
    build_stock_cohorts,
    cohort_event_metrics,
    censoring_diagnostics,
    simulate_daily_ops,
    phase_table,
    event_recall_by_event_date,
)
from src.eval.report import write_run_outputs, pretty_print_dict


logger = get_logger("run_eval")

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/config.yaml")
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--mode", choices=["threshold", "topk"], default="threshold")
    ap.add_argument("--k", type=int, default=50, help="Top-K for daily policy when mode=topk")
    args = ap.parse_args()

    cfg       = load_yaml(Path(args.cfg))
    paths_cfg = load_yaml(Path(args.paths))
    paths = resolve_paths(paths_cfg, project_root=Path.cwd())

    run_dir = paths.outputs_dir / args.run_id
    ensure_dir(run_dir / "eval")

    preds_path = run_dir / "preds" / f"{args.split}_preds.parquet"
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing predictions: {preds_path}. Run scripts/run_predict.py first.")
    eval_df = _read_df(preds_path)
    eval_df = validate_eval_df(eval_df)

    # Load events
    DIV_DISTCD = cfg['div_distcd']
    events_raw = load_div_events(paths.tableA_path, DIV_DISTCD=DIV_DISTCD, permno_col="PERMNO", event_date_col="DCLRDT")
    events_df = standardize_events(events_raw, permno_col="PERMNO", event_date_col="DCLRDT")

    eligible = filter_events_for_eval(eval_df, events_df)

    H = int(cfg.get("H_eval", cfg.get("H_label", 10)))
    threshold = float(cfg.get("threshold", 0.5))
    cooldown_td = int(cfg.get("cooldown_td", 0))

    # Global metrics (row-level)
    gm = global_metrics(eval_df, threshold=threshold)

    if args.mode == "threshold":
        events_out, alerts_out, summary = event_level_report_v2(
            eval_df=eval_df,
            events_df=eligible,
            H=H,
            threshold=threshold,
            cooldown_td=cooldown_td,
            censoring_mode="exclude",
        )
    else:
        alerts = make_daily_topk_alerts(eval_df, k=args.k)
        events_out, alerts_out, summary = evaluate_alerts_forward_window(
            eval_df=eval_df,
            events_df=eligible,
            alerts_df=alerts,
            H=H,
            censoring_mode="exclude",
        )
        summary["policy"] = f"daily_topk(k={args.k})"

    # Daily top-k report always useful (diagnostic); use k from config if present
    k0 = int(cfg.get("topk_list", [50])[0])
    daily = daily_topk_report(eval_df, k=k0)

    # Cadence & cohorts
    cutoff_date = str(eval_df["date"].min())
    cadence_cutoff = compute_cadence_stats(events_df, cutoff_date=cutoff_date)
    cadence_full = compute_cadence_stats(events_df, cutoff_date=None)

    cohorts = build_stock_cohorts(eval_df, cadence_cutoff)
    cohorts_full = build_stock_cohorts(eval_df, cadence_full)

    stock_best_worst = stock_aucpr_best_worst(
        eval_df, stock_cohorts_cutoff=cohorts, stock_cohorts_full=cohorts_full, top_n=10
    )

    cohort_cols = [
        "bucket_n_events",
        "bucket_gap_cv",
        "bucket_size",
        "bucket_liquidity",
        # "industry",
        "bucket_vol_regime",
        "is_quarterly_clockwork",
    ]
    cohorts_rep = cohort_event_metrics(eval_df, events_out, alerts_out, cohorts, cohort_cols, censoring_mode="exclude")

    censor_diag = censoring_diagnostics(alerts_out)
    ops_daily = simulate_daily_ops(eval_df, alerts_out, events_out)
    phase_tab = phase_table(events_out, H=H)
    er_daily = event_recall_by_event_date(events_out)

    # Aggregate summary
    full_summary: Dict[str, Any] = {"global_metrics": gm, "event_level_summary": summary}

    logger.info(pretty_print_dict(gm, "Global Metrics"))
    logger.info(pretty_print_dict(summary, "Event-level Summary"))

    write_run_outputs(
        run_dir=run_dir,
        eval_df=eval_df,
        events_out=events_out,
        alerts_out=alerts_out,
        daily_topk=daily,
        cohorts_report=cohorts_rep,
        censoring_diag=censor_diag,
        phase_tab=phase_tab,
        summary=full_summary,
        stock_aucpr_best_worst=stock_best_worst,
    )

    ops_daily.to_csv(run_dir / "eval" / "ops_daily.csv", index=False)
    er_daily.to_csv(run_dir / "eval" / "event_recall_by_date.csv", index=False)
    cohorts.to_csv(run_dir / "eval" / "stock_cohorts.csv", index=False)
    cohorts.to_csv(run_dir / "eval" / "stock_cohorts_cutoff.csv", index=False)
    cohorts_full.to_csv(run_dir / "eval" / "stock_cohorts_full.csv", index=False)

    logger.info(f"wrote eval outputs under: {run_dir/'eval'}")


if __name__ == "__main__":
    main()
