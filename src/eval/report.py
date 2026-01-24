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
