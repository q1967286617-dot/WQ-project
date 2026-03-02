from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.utils.paths import load_yaml, resolve_paths, ensure_dir
from src.utils.logging import get_logger
from src.data.load import load_div_events, standardize_events
from src.eval.eval_tools import (
    validate_eval_df,
    filter_events_for_eval,
    evaluate_alerts_forward_window,
    generate_alerts_threshold,
)


logger = get_logger("run_threshold_sweep")


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


def _parse_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


def _make_range(start: float, end: float, step: float) -> List[float]:
    vals = np.arange(start, end + 1e-9, step).tolist()
    return [round(float(v), 6) for v in vals]


def _attach_group_col(
    eval_df: pd.DataFrame,
    split_df: pd.DataFrame,
    group_col: str,
    date_col: str,
    permno_col: str,
) -> pd.DataFrame:
    if group_col in eval_df.columns:
        return eval_df
    if group_col not in split_df.columns:
        raise ValueError(f"split 数据中找不到分层列 {group_col}")

    x = split_df[[date_col, permno_col, group_col]].copy()
    x[date_col] = pd.to_datetime(x[date_col], errors="coerce")
    x[permno_col] = x[permno_col].astype(int)

    out = eval_df.merge(
        x,
        left_on=["date", "permno"],
        right_on=[date_col, permno_col],
        how="left",
    ).drop(columns=[date_col, permno_col])

    if out[group_col].isna().all():
        raise ValueError(f"{group_col} 合并失败，结果全是空值")
    return out


def _global_metrics_with_row_threshold(
    eval_df: pd.DataFrame,
    thresholds: Dict[int, float],
    group_col: str,
) -> Dict[str, float]:
    df = validate_eval_df(eval_df)
    if group_col not in df.columns:
        raise ValueError(f"eval_df 缺少分层列: {group_col}")

    g = df[group_col].fillna(0).astype(int).values
    p = df["prob"].values
    y = df["y"].values

    thr = np.where(g >= 1, thresholds.get(1, 0.5), thresholds.get(0, 0.5))
    pred = (p >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )

    return {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "alert_rate": float(pred.mean()),
        "fpr": float(fp / (fp + tn + 1e-12)),
    }


def _generate_alerts_threshold_by_group(
    eval_df: pd.DataFrame,
    thresholds: Dict[int, float],
    group_col: str,
    cooldown_td: int = 0,
) -> pd.DataFrame:
    df = validate_eval_df(eval_df)
    if group_col not in df.columns:
        raise ValueError(f"eval_df 缺少分层列: {group_col}")

    rows: List[Dict[str, Any]] = []
    for permno, g in df.groupby("permno", sort=False):
        probs = g["prob"].values
        dates = g["date"].values
        groups = g[group_col].fillna(0).astype(int).values
        last_alert_i = -10**18

        for i in range(len(g)):
            thr = thresholds.get(1, 0.5) if groups[i] >= 1 else thresholds.get(0, 0.5)
            if probs[i] >= thr:
                if cooldown_td > 0 and i <= last_alert_i + cooldown_td:
                    continue
                rows.append({
                    "permno": int(permno),
                    "date": pd.to_datetime(dates[i]),
                    "prob": float(probs[i]),
                    "aidx": int(i),
                })
                last_alert_i = i

    return pd.DataFrame(rows).sort_values(["permno", "date"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/config.yaml")
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--model_cfg", default="configs/model.yaml")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--group_col", default="has_div_history")

    ap.add_argument("--t_hist_list", default=None)
    ap.add_argument("--t_nohist_list", default=None)
    ap.add_argument("--t_hist_start", type=float, default=0.60)
    ap.add_argument("--t_hist_end", type=float, default=0.90)
    ap.add_argument("--t_hist_step", type=float, default=0.02)
    ap.add_argument("--t_nohist_start", type=float, default=0.75)
    ap.add_argument("--t_nohist_end", type=float, default=0.95)
    ap.add_argument("--t_nohist_step", type=float, default=0.02)

    ap.add_argument("--cooldown_list", default="10")
    ap.add_argument("--max_hit_drop", type=float, default=0.01)
    ap.add_argument("--enforce_nohist_ge_hist", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(Path(args.cfg))
    paths_cfg = load_yaml(Path(args.paths))
    model_cfg = load_yaml(Path(args.model_cfg))
    paths = resolve_paths(paths_cfg, project_root=Path.cwd())

    run_dir = paths.outputs_dir / args.run_id
    preds_path = run_dir / "preds" / f"{args.split}_preds.parquet"
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing predictions: {preds_path}. Run scripts/run_predict.py first.")
    eval_df = _read_df(preds_path)
    eval_df = validate_eval_df(eval_df)

    # Attach group column if missing (merge from processed split)
    date_col = model_cfg.get("date_col", "DlyCalDt")
    permno_col = model_cfg.get("permno_col", "PERMNO")
    split_path = paths.processed_dir / f"{args.split}.parquet"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    split_df = _read_df(split_path)
    eval_df = _attach_group_col(eval_df, split_df, args.group_col, date_col, permno_col)

    # Load events
    DIV_DISTCD = cfg["div_distcd"]
    events_raw = load_div_events(paths.tableA_path, DIV_DISTCD=DIV_DISTCD, permno_col="PERMNO", event_date_col="DCLRDT")
    events_df = standardize_events(events_raw, permno_col="PERMNO", event_date_col="DCLRDT")
    eligible = filter_events_for_eval(eval_df, events_df)

    H = int(cfg.get("H_eval", cfg.get("H_label", 10)))
    base_threshold = float(cfg.get("threshold", 0.5))
    base_cooldown = int(cfg.get("cooldown_td", 0))

    # Baseline (single threshold from config)
    base_alerts = generate_alerts_threshold(eval_df, threshold=base_threshold, cooldown_td=base_cooldown)
    _, _, base_summary = evaluate_alerts_forward_window(
        eval_df=eval_df,
        events_df=eligible,
        alerts_df=base_alerts,
        H=H,
        censoring_mode="exclude",
    )
    base_hit = float(base_summary.get("hit_rate_events", np.nan))

    if args.t_hist_list:
        t_hist_list = _parse_list(args.t_hist_list)
    else:
        t_hist_list = _make_range(args.t_hist_start, args.t_hist_end, args.t_hist_step)

    if args.t_nohist_list:
        t_nohist_list = _parse_list(args.t_nohist_list)
    else:
        t_nohist_list = _make_range(args.t_nohist_start, args.t_nohist_end, args.t_nohist_step)

    cooldown_list = [int(x) for x in _parse_list(args.cooldown_list)]

    rows = []
    for cooldown_td in cooldown_list:
        for t_hist in t_hist_list:
            for t_nohist in t_nohist_list:
                if args.enforce_nohist_ge_hist and t_nohist < t_hist:
                    continue
                thresholds = {1: float(t_hist), 0: float(t_nohist)}
                alerts = _generate_alerts_threshold_by_group(
                    eval_df=eval_df,
                    thresholds=thresholds,
                    group_col=args.group_col,
                    cooldown_td=cooldown_td,
                )
                _, _, summary = evaluate_alerts_forward_window(
                    eval_df=eval_df,
                    events_df=eligible,
                    alerts_df=alerts,
                    H=H,
                    censoring_mode="exclude",
                )
                gm = _global_metrics_with_row_threshold(
                    eval_df=eval_df,
                    thresholds=thresholds,
                    group_col=args.group_col,
                )

                rows.append({
                    "cooldown_td": cooldown_td,
                    "t_hist": float(t_hist),
                    "t_nohist": float(t_nohist),
                    "hit_rate_events": float(summary.get("hit_rate_events", np.nan)),
                    "false_alerts_per_1000_stockdays": float(summary.get("false_alerts_per_1000_stockdays", np.nan)),
                    "false_alert_rate": float(summary.get("false_alert_rate", np.nan)),
                    "n_alerts_eval": int(summary.get("n_alerts_eval", 0)),
                    "hit_events": int(summary.get("hit_events", 0)),
                    "precision": gm["precision"],
                    "recall": gm["recall"],
                    "f1": gm["f1"],
                    "alert_rate": gm["alert_rate"],
                    "fpr": gm["fpr"],
                })

    results = pd.DataFrame(rows)
    results = results.sort_values(
        ["false_alerts_per_1000_stockdays", "hit_rate_events"],
        ascending=[True, False],
    ).reset_index(drop=True)

    # Recommend by constraint: hit_rate >= baseline - max_hit_drop
    min_hit = base_hit - float(args.max_hit_drop) if not np.isnan(base_hit) else -np.inf
    feasible = results[results["hit_rate_events"] >= min_hit].copy()
    if len(feasible) == 0:
        feasible = results.copy()
    best = feasible.sort_values(
        ["false_alerts_per_1000_stockdays", "hit_rate_events"],
        ascending=[True, False],
    ).head(1)

    out_dir = run_dir / "eval" / "threshold_sweep"
    ensure_dir(out_dir)
    out_csv = out_dir / f"threshold_sweep_{args.split}.csv"
    results.to_csv(out_csv, index=False)

    best_path = out_dir / f"best_{args.split}.json"
    best_row = best.iloc[0].to_dict()
    best_row["baseline_hit_rate_events"] = base_hit
    best_row["baseline_threshold"] = base_threshold
    best_row["baseline_cooldown_td"] = base_cooldown
    pd.Series(best_row).to_json(best_path)

    logger.info(f"baseline hit_rate_events={base_hit:.6f} (threshold={base_threshold}, cooldown={base_cooldown})")
    logger.info(f"saved sweep results: {out_csv}")
    logger.info(f"best config: {best_row}")


if __name__ == "__main__":
    main()
