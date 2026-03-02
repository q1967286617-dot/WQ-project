from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
from typing import Any, Dict

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
)
from src.eval.report import dump_df, dump_json, pretty_print_dict


logger = get_logger("run_eval_stratified")


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
    t_hist: float,
    t_nohist: float,
    group_col: str,
) -> Dict[str, float]:
    df = validate_eval_df(eval_df)
    if group_col not in df.columns:
        raise ValueError(f"eval_df 缺少分层列: {group_col}")

    g = df[group_col].fillna(0).astype(int).values
    p = df["prob"].values
    y = df["y"].values

    thr = np.where(g >= 1, t_hist, t_nohist)
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
    t_hist: float,
    t_nohist: float,
    group_col: str,
    cooldown_td: int = 0,
) -> pd.DataFrame:
    df = validate_eval_df(eval_df)
    if group_col not in df.columns:
        raise ValueError(f"eval_df 缺少分层列: {group_col}")

    rows = []
    for permno, g in df.groupby("permno", sort=False):
        probs = g["prob"].values
        dates = g["date"].values
        groups = g[group_col].fillna(0).astype(int).values
        last_alert_i = -10**18

        for i in range(len(g)):
            thr = t_hist if groups[i] >= 1 else t_nohist
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
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--group_col", default="has_div_history")
    ap.add_argument("--t_hist", type=float, required=True)
    ap.add_argument("--t_nohist", type=float, required=True)
    ap.add_argument("--cooldown_td", type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml(Path(args.cfg))
    paths_cfg = load_yaml(Path(args.paths))
    model_cfg = load_yaml(Path(args.model_cfg))
    paths = resolve_paths(paths_cfg, project_root=Path.cwd())

    run_dir = paths.outputs_dir / args.run_id
    out_dir = run_dir / "eval_stratified"
    ensure_dir(out_dir)

    preds_path = run_dir / "preds" / f"{args.split}_preds.parquet"
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing predictions: {preds_path}. Run scripts/run_predict.py first.")
    eval_df = _read_df(preds_path)
    eval_df = validate_eval_df(eval_df)

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
    cooldown_td = int(cfg.get("cooldown_td", 0)) if args.cooldown_td is None else int(args.cooldown_td)

    alerts = _generate_alerts_threshold_by_group(
        eval_df=eval_df,
        t_hist=float(args.t_hist),
        t_nohist=float(args.t_nohist),
        group_col=args.group_col,
        cooldown_td=cooldown_td,
    )
    events_out, alerts_out, summary = evaluate_alerts_forward_window(
        eval_df=eval_df,
        events_df=eligible,
        alerts_df=alerts,
        H=H,
        censoring_mode="exclude",
    )

    gm = _global_metrics_with_row_threshold(
        eval_df=eval_df,
        t_hist=float(args.t_hist),
        t_nohist=float(args.t_nohist),
        group_col=args.group_col,
    )

    full_summary: Dict[str, Any] = {
        "policy": f"stratified_threshold(group={args.group_col}, t_hist={args.t_hist}, t_nohist={args.t_nohist}, cooldown={cooldown_td})",
        "global_metrics": gm,
        "event_level_summary": summary,
    }

    logger.info(pretty_print_dict(gm, "Global Metrics (Stratified)"))
    logger.info(pretty_print_dict(summary, "Event-level Summary (Stratified)"))

    dump_df(events_out, out_dir / "events_out.csv")
    dump_df(alerts_out, out_dir / "alerts_out.csv")
    dump_json(full_summary, out_dir / "summary.json")
    logger.info(f"wrote stratified eval outputs under: {out_dir}")


if __name__ == "__main__":
    main()
