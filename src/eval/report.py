from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, List

import pandas as pd

from ..utils.paths import ensure_dir


def dump_json(obj: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def dump_df(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    if path.suffix.lower() in {".parquet"}:
        try:
            df.to_parquet(path, index=False)
        except Exception:
            # Parquet engine not installed; fallback to CSV
            df.to_csv(path.with_suffix(".csv"), index=False)
    else:
        df.to_csv(path, index=False)


def pretty_print_dict(d: Dict[str, Any], title: str) -> str:
    lines = [f"===== {title} ====="]
    for k, v in d.items():
        lines.append(f"{k:>22}: {v}")
    return "\n".join(lines)


def _series_stats(s: pd.Series) -> Dict[str, Any]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {}
    q = s.quantile([0.25, 0.5, 0.75])
    return {
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "min": float(s.min()),
        "p25": float(q.loc[0.25]),
        "p50": float(q.loc[0.5]),
        "p75": float(q.loc[0.75]),
        "max": float(s.max()),
    }


def _series_stats_with_unknown(s: pd.Series) -> Dict[str, Any]:
    n_total = int(len(s))
    s_num = pd.to_numeric(s, errors="coerce")
    n_known = int(s_num.notna().sum())
    n_unknown = int(n_total - n_known)
    out: Dict[str, Any] = {
        "n_total": n_total,
        "n_known": n_known,
        "n_unknown": n_unknown,
        "unknown_rate": float(n_unknown / (n_total + 1e-12)) if n_total else None,
        "stats": _series_stats(s_num),
    }
    return out


def _summarize_best_worst_full(stock_aucpr_best_worst: pd.DataFrame) -> Dict[str, Any]:
    """
    Summarize best/worst stocks using full-history cohort fields only,
    and provide a simple best-vs-worst comparison.
    """
    df = stock_aucpr_best_worst.copy()
    if df.empty or "rank_group" not in df.columns:
        return {}

    full_fields = [c for c in ["gap_cv_full", "gap_med_full", "div_count_full", "n_events_full"] if c in df.columns]
    core_fields = [c for c in ["aucpr", "auc", "n", "n_pos", "pos_rate"] if c in df.columns]
    fields = core_fields + full_fields

    out: Dict[str, Any] = {}
    means: Dict[str, Dict[str, float]] = {}

    for group in ["best", "worst"]:
        g = df[df["rank_group"] == group].copy()
        if g.empty:
            continue

        group_summary: Dict[str, Any] = {"n_stocks": int(g["permno"].nunique()) if "permno" in g.columns else int(len(g))}
        field_stats: Dict[str, Any] = {}
        mean_map: Dict[str, float] = {}

        for c in fields:
            stats = _series_stats_with_unknown(g[c])
            field_stats[c] = stats

            s_num = pd.to_numeric(g[c], errors="coerce")
            if s_num.notna().any():
                mean_map[c] = float(s_num.mean())

        group_summary["fields"] = field_stats
        out[group] = group_summary
        means[group] = mean_map

    # Best vs worst comparison on the full-history fields.
    compare_rows: List[Dict[str, Any]] = []
    for c in full_fields:
        mb = means.get("best", {}).get(c)
        mw = means.get("worst", {}).get(c)
        if mb is None or mw is None:
            compare_rows.append({"field": c, "best_mean": mb, "worst_mean": mw})
            continue

        compare_rows.append(
            {
                "field": c,
                "best_mean": mb,
                "worst_mean": mw,
                "best_minus_worst": float(mb - mw),
                "best_over_worst": float(mb / (mw + 1e-12)),
            }
        )

    out["best_vs_worst_full_fields"] = compare_rows
    return out


def _build_analysis_summary(
    eval_df: Optional[pd.DataFrame],
    events_out: Optional[pd.DataFrame],
    alerts_out: Optional[pd.DataFrame],
    daily_topk: Optional[pd.DataFrame],
    cohorts_report: Optional[pd.DataFrame],
    censoring_diag: Optional[pd.DataFrame],
    phase_tab: Optional[pd.DataFrame],
    summary: Optional[Dict[str, Any]],
    stock_aucpr_best_worst: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    """
    Lightweight, structured analysis over the main eval outputs.
    This avoids re-reading CSVs and keeps changes localized to report.py.
    """
    out: Dict[str, Any] = {}

    # 0) Preserve existing high-level metrics if provided.
    if summary:
        out["from_eval_summary"] = summary

    # 1) Eval dataframe overview.
    if eval_df is not None and not eval_df.empty:
        df = eval_df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        overview: Dict[str, Any] = {
            "n_rows": int(len(df)),
            "n_permnos": int(df["permno"].nunique()) if "permno" in df.columns else None,
        }
        if "date" in df.columns and df["date"].notna().any():
            overview["date_min"] = str(df["date"].min())
            overview["date_max"] = str(df["date"].max())
        if "y" in df.columns:
            overview["base_pos_rate"] = float(pd.to_numeric(df["y"], errors="coerce").mean())
        if "prob" in df.columns:
            overview["prob_stats"] = _series_stats(df["prob"])

        out["eval_overview"] = overview

    # 2) Alerts-level overview.
    if alerts_out is not None and not alerts_out.empty:
        a = alerts_out.copy()
        if "date" in a.columns:
            a["date"] = pd.to_datetime(a["date"], errors="coerce")

        n_total = int(len(a))
        is_censored = (
            pd.to_numeric(a["is_censored"], errors="coerce").fillna(0).astype(int)
            if "is_censored" in a.columns
            else pd.Series(0, index=a.index, dtype=int)
        )
        n_censored = int(is_censored.sum())
        a_eval = a[is_censored == 0].copy()
        n_eval = int(len(a_eval))

        is_true_alert = (
            pd.to_numeric(a_eval["is_true_alert"], errors="coerce").fillna(0).astype(int)
            if "is_true_alert" in a_eval.columns
            else pd.Series(0, index=a_eval.index, dtype=int)
        )

        true_alerts = int((is_true_alert == 1).sum()) if n_eval else 0
        false_alerts = int((is_true_alert == 0).sum()) if n_eval else 0

        alerts_summary: Dict[str, Any] = {
            "n_alerts_total": n_total,
            "n_censored": n_censored,
            "n_alerts_eval": n_eval,
            "true_alerts": true_alerts,
            "false_alerts": false_alerts,
            "true_alert_rate": float(true_alerts / (n_eval + 1e-12)) if n_eval else None,
            "false_alert_rate": float(false_alerts / (n_eval + 1e-12)) if n_eval else None,
        }
        if "prob" in a_eval.columns:
            alerts_summary["prob_stats_on_eval_alerts"] = _series_stats(a_eval["prob"])
        if "lead_time_td" in a_eval.columns:
            lead = pd.to_numeric(
                a_eval.loc[is_true_alert == 1, "lead_time_td"], errors="coerce"
            )
            alerts_summary["lead_time_stats_on_true_alerts"] = _series_stats(lead)

        out["alerts_overview"] = alerts_summary

    # 3) Events-level overview.
    if events_out is not None and not events_out.empty:
        e = events_out.copy()
        n_events = int(len(e))
        hit_series = (
            pd.to_numeric(e["hit"], errors="coerce").fillna(0).astype(int)
            if "hit" in e.columns
            else pd.Series(0, index=e.index, dtype=int)
        )
        hit_events = int(hit_series.sum())
        events_summary: Dict[str, Any] = {
            "n_events": n_events,
            "hit_events": hit_events,
            "hit_rate": float(hit_events / (n_events + 1e-12)) if n_events else None,
        }
        if "best_lead_time_td" in e.columns:
            lead_hit = pd.to_numeric(
                e.loc[hit_series == 1, "best_lead_time_td"], errors="coerce"
            )
            events_summary["best_lead_time_stats_on_hits"] = _series_stats(lead_hit)
        out["events_overview"] = events_summary

    # 4) Daily Top-K summary (by k).
    if daily_topk is not None and not daily_topk.empty and "k" in daily_topk.columns:
        d = daily_topk.copy()
        if "date" in d.columns:
            d["date"] = pd.to_datetime(d["date"], errors="coerce")

        topk_summary = (
            d.groupby("k")
            .agg(
                n_days=("date", "nunique") if "date" in d.columns else ("k", "size"),
                precision_at_k_mean=("precision_at_k", "mean"),
                precision_at_k_median=("precision_at_k", "median"),
                recall_at_k_mean=("recall_at_k", "mean"),
                recall_at_k_median=("recall_at_k", "median"),
                lift_at_k_mean=("lift_at_k", "mean"),
                lift_at_k_median=("lift_at_k", "median"),
            )
            .reset_index()
            .sort_values("k")
        )
        out["daily_topk_overview"] = topk_summary.to_dict(orient="records")

    # 5) Cohort diagnostics: surface worst buckets by false alert rate.
    if cohorts_report is not None and not cohorts_report.empty:
        c = cohorts_report.copy()
        if {"cohort_dim", "bucket", "false_alert_rate"}.issubset(c.columns):
            worst_rows: List[Dict[str, Any]] = []
            for dim, g in c.groupby("cohort_dim", dropna=False):
                g2 = g.sort_values("false_alert_rate", ascending=False).head(3)
                for _, r in g2.iterrows():
                    worst_rows.append(
                        {
                            "cohort_dim": dim,
                            "bucket": r.get("bucket"),
                            "n_events": r.get("n_events"),
                            "hit_rate_events": r.get("hit_rate_events"),
                            "false_alert_rate": r.get("false_alert_rate"),
                        }
                    )
            out["cohort_worst_buckets_by_false_alert_rate"] = worst_rows

    # 6) Censoring diagnostics summary.
    if censoring_diag is not None and not censoring_diag.empty:
        cd = censoring_diag.copy()
        if {"n_alerts", "n_censored"}.issubset(cd.columns):
            n_alerts = float(pd.to_numeric(cd["n_alerts"], errors="coerce").fillna(0).sum())
            n_cens = float(pd.to_numeric(cd["n_censored"], errors="coerce").fillna(0).sum())
            censor_summary: Dict[str, Any] = {
                "n_alerts": int(n_alerts),
                "n_censored": int(n_cens),
                "censored_rate": float(n_cens / (n_alerts + 1e-12)) if n_alerts else None,
            }
            if "censored_rate" in cd.columns and "month" in cd.columns:
                worst_months = (
                    cd.sort_values("censored_rate", ascending=False)
                    .head(3)[["month", "n_alerts", "n_censored", "censored_rate"]]
                )
                censor_summary["worst_months"] = worst_months.to_dict(orient="records")
            out["censoring_overview"] = censor_summary

    # 7) Phase / lead-time buckets, if present.
    if phase_tab is not None and not phase_tab.empty:
        out["phase_table"] = phase_tab.to_dict(orient="records")

    # 8) Best/worst stocks by AUC-PR, if present.
    if stock_aucpr_best_worst is not None and not stock_aucpr_best_worst.empty:
        out["stock_aucpr_best_worst"] = stock_aucpr_best_worst.to_dict(orient="records")
        out["stock_aucpr_best_worst_full_summary"] = _summarize_best_worst_full(stock_aucpr_best_worst)

    return out


def write_run_outputs(
    run_dir: Path,
    eval_df: Optional[pd.DataFrame] = None,
    events_out: Optional[pd.DataFrame] = None,
    alerts_out: Optional[pd.DataFrame] = None,
    daily_topk: Optional[pd.DataFrame] = None,
    cohorts_report: Optional[pd.DataFrame] = None,
    censoring_diag: Optional[pd.DataFrame] = None,
    phase_tab: Optional[pd.DataFrame] = None,
    summary: Optional[Dict[str, Any]] = None,
    stock_aucpr_best_worst: Optional[pd.DataFrame] = None,
) -> None:
    ensure_dir(run_dir)

    if eval_df is not None:
        dump_df(eval_df, run_dir / "preds" / "eval_df.parquet")
    if events_out is not None:
        dump_df(events_out, run_dir / "eval" / "events_out.csv")
    if alerts_out is not None:
        dump_df(alerts_out, run_dir / "eval" / "alerts_out.csv")
    if daily_topk is not None:
        dump_df(daily_topk, run_dir / "eval" / "daily_topk.csv")
    if cohorts_report is not None:
        dump_df(cohorts_report, run_dir / "eval" / "cohorts.csv")
    if censoring_diag is not None:
        dump_df(censoring_diag, run_dir / "eval" / "censoring_diag.csv")
    if phase_tab is not None:
        dump_df(phase_tab, run_dir / "eval" / "phase_table.csv")
    if summary is not None:
        dump_json(summary, run_dir / "eval" / "summary.json")
    if stock_aucpr_best_worst is not None and not stock_aucpr_best_worst.empty:
        dump_df(stock_aucpr_best_worst, run_dir / "eval" / "stock_aucpr_best_worst.csv")

    # Minimal extra: write a structured, analysis-focused summary json.
    analysis_summary = _build_analysis_summary(
        eval_df=eval_df,
        events_out=events_out,
        alerts_out=alerts_out,
        daily_topk=daily_topk,
        cohorts_report=cohorts_report,
        censoring_diag=censoring_diag,
        phase_tab=phase_tab,
        summary=summary,
        stock_aucpr_best_worst=stock_aucpr_best_worst,
    )
    if analysis_summary:
        dump_json(analysis_summary, run_dir / "eval" / "analysis_summary.json")
