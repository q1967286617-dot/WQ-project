from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.append(str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.data.load import load_div_events, standardize_events
from src.eval.eval_tools import (
    evaluate_alerts_forward_window,
    filter_events_for_eval,
    validate_eval_df,
)
from src.eval.report import dump_df, dump_json, pretty_print_dict
from src.utils.logging import get_logger
from src.utils.paths import ensure_dir, load_yaml, resolve_paths


logger = get_logger("run_eval_stratified")


def _read_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read parquet: {path}. Install pyarrow (recommended) or convert to CSV.\n"
                f"Original error: {exc}"
            )
    return pd.read_csv(path)


def _resolve_preds_path(run_dir: Path, split: str) -> Path:
    parquet_path = run_dir / "preds" / f"{split}_preds.parquet"
    csv_path = run_dir / "preds" / f"{split}_preds.csv"
    if parquet_path.exists():
        return parquet_path
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError(
        f"Missing predictions for split={split}. Expected one of:\n"
        f"  {parquet_path}\n"
        f"  {csv_path}\n"
        "Run scripts/run_predict.py first."
    )


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
        raise ValueError(f"Missing group column in split dataframe: {group_col}")

    keys = split_df[[date_col, permno_col, group_col]].copy()
    keys[date_col] = pd.to_datetime(keys[date_col], errors="coerce")
    keys[permno_col] = keys[permno_col].astype(int)

    out = eval_df.merge(
        keys,
        left_on=["date", "permno"],
        right_on=[date_col, permno_col],
        how="left",
    ).drop(columns=[date_col, permno_col])

    if out[group_col].isna().all():
        raise ValueError(f"Failed to attach group column: {group_col}")
    return out


def _load_eval_df_for_split(
    run_dir: Path,
    paths,
    split: str,
    group_col: str,
    date_col: str,
    permno_col: str,
) -> pd.DataFrame:
    preds_path = _resolve_preds_path(run_dir, split)
    eval_df = validate_eval_df(_read_df(preds_path))

    split_path = paths.processed_dir / f"{split}.parquet"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    split_df = _read_df(split_path)
    return _attach_group_col(eval_df, split_df, group_col, date_col, permno_col)


def _best_threshold_by_f1(y: np.ndarray, prob: np.ndarray, grid_size: int = 201) -> Tuple[float, float]:
    if len(y) == 0:
        return 0.5, 0.0

    y = y.astype(int)
    prob = prob.astype(float)
    pos = int(y.sum())

    if pos == 0:
        return 1.0, 0.0
    if pos == len(y):
        return 0.0, 1.0

    thresholds = np.linspace(0.0, 1.0, grid_size)
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        pred = (prob >= threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y, pred, average="binary", zero_division=0
        )
        if (f1 > best_f1 + 1e-12) or (
            abs(f1 - best_f1) <= 1e-12 and threshold > best_threshold
        ):
            best_threshold = float(threshold)
            best_f1 = float(f1)

    return best_threshold, best_f1


def _select_thresholds_on_val(eval_df: pd.DataFrame, group_col: str) -> Dict[str, float]:
    df = validate_eval_df(eval_df)
    if group_col not in df.columns:
        raise ValueError(f"Missing group column in eval dataframe: {group_col}")

    groups = df[group_col].fillna(0).astype(int)
    hist_mask = groups >= 1
    nohist_mask = ~hist_mask

    t_hist, f1_hist = _best_threshold_by_f1(
        df.loc[hist_mask, "y"].to_numpy(),
        df.loc[hist_mask, "prob"].to_numpy(),
    )
    t_nohist, f1_nohist = _best_threshold_by_f1(
        df.loc[nohist_mask, "y"].to_numpy(),
        df.loc[nohist_mask, "prob"].to_numpy(),
    )

    return {
        "t_hist": float(t_hist),
        "t_nohist": float(t_nohist),
        "f1_hist": float(f1_hist),
        "f1_nohist": float(f1_nohist),
        "n_hist": int(hist_mask.sum()),
        "n_nohist": int(nohist_mask.sum()),
    }


def _global_metrics_with_row_threshold(
    eval_df: pd.DataFrame,
    t_hist: float,
    t_nohist: float,
    group_col: str,
) -> Dict[str, float]:
    df = validate_eval_df(eval_df)
    if group_col not in df.columns:
        raise ValueError(f"Missing group column in eval dataframe: {group_col}")

    groups = df[group_col].fillna(0).astype(int).values
    prob = df["prob"].values
    y = df["y"].values

    thresholds = np.where(groups >= 1, t_hist, t_nohist)
    pred = (prob >= thresholds).astype(int)

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
        raise ValueError(f"Missing group column in eval dataframe: {group_col}")

    rows = []
    for permno, group in df.groupby("permno", sort=False):
        probs = group["prob"].values
        dates = group["date"].values
        groups = group[group_col].fillna(0).astype(int).values
        last_alert_i = -10**18

        for i in range(len(group)):
            threshold = t_hist if groups[i] >= 1 else t_nohist
            if probs[i] >= threshold:
                if cooldown_td > 0 and i <= last_alert_i + cooldown_td:
                    continue
                rows.append(
                    {
                        "permno": int(permno),
                        "date": pd.to_datetime(dates[i]),
                        "prob": float(probs[i]),
                        "aidx": int(i),
                    }
                )
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
    ap.add_argument("--t_hist", type=float, default=None)
    ap.add_argument("--t_nohist", type=float, default=None)
    ap.add_argument("--cooldown_td", type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml(Path(args.cfg))
    paths_cfg = load_yaml(Path(args.paths))
    model_cfg = load_yaml(Path(args.model_cfg))
    paths = resolve_paths(paths_cfg, project_root=Path.cwd())

    run_dir = paths.outputs_dir / args.run_id
    out_dir = run_dir / "eval_stratified"
    ensure_dir(out_dir)

    date_col = model_cfg.get("date_col", "DlyCalDt")
    permno_col = model_cfg.get("permno_col", "PERMNO")

    eval_df = _load_eval_df_for_split(
        run_dir=run_dir,
        paths=paths,
        split=args.split,
        group_col=args.group_col,
        date_col=date_col,
        permno_col=permno_col,
    )

    selected = None
    if args.t_hist is None or args.t_nohist is None:
        val_eval_df = _load_eval_df_for_split(
            run_dir=run_dir,
            paths=paths,
            split="val",
            group_col=args.group_col,
            date_col=date_col,
            permno_col=permno_col,
        )
        selected = _select_thresholds_on_val(val_eval_df, args.group_col)

    t_hist = float(args.t_hist) if args.t_hist is not None else float(selected["t_hist"])
    t_nohist = float(args.t_nohist) if args.t_nohist is not None else float(selected["t_nohist"])

    if selected is not None:
        threshold_msg = (
            "Auto-selected thresholds on val: "
            f"t_hist={t_hist:.4f} (best_f1={selected['f1_hist']:.4f}, n={selected['n_hist']}), "
            f"t_nohist={t_nohist:.4f} (best_f1={selected['f1_nohist']:.4f}, n={selected['n_nohist']})"
        )
    else:
        threshold_msg = f"Using provided thresholds: t_hist={t_hist:.4f}, t_nohist={t_nohist:.4f}"

    print(threshold_msg)
    logger.info(threshold_msg)

    div_distcd = cfg["div_distcd"]
    events_raw = load_div_events(
        paths.tableA_path,
        div_distcd=div_distcd,
        permno_col="PERMNO",
        event_date_col="DCLRDT",
    )
    events_df = standardize_events(events_raw, permno_col="PERMNO", event_date_col="DCLRDT")
    eligible = filter_events_for_eval(eval_df, events_df)

    horizon = int(cfg.get("H_eval", cfg.get("H_label", 10)))
    cooldown_td = int(cfg.get("cooldown_td", 0)) if args.cooldown_td is None else int(args.cooldown_td)

    alerts = _generate_alerts_threshold_by_group(
        eval_df=eval_df,
        t_hist=t_hist,
        t_nohist=t_nohist,
        group_col=args.group_col,
        cooldown_td=cooldown_td,
    )
    events_out, alerts_out, summary = evaluate_alerts_forward_window(
        eval_df=eval_df,
        events_df=eligible,
        alerts_df=alerts,
        H=horizon,
        censoring_mode="exclude",
    )

    gm = _global_metrics_with_row_threshold(
        eval_df=eval_df,
        t_hist=t_hist,
        t_nohist=t_nohist,
        group_col=args.group_col,
    )

    full_summary: Dict[str, Any] = {
        "policy": (
            f"stratified_threshold(group={args.group_col}, "
            f"t_hist={t_hist}, t_nohist={t_nohist}, cooldown={cooldown_td})"
        ),
        "global_metrics": gm,
        "event_level_summary": summary,
    }
    if selected is not None:
        full_summary["selected_thresholds_on_val"] = selected

    logger.info(pretty_print_dict(gm, "Global Metrics (Stratified)"))
    logger.info(pretty_print_dict(summary, "Event-level Summary (Stratified)"))

    dump_df(events_out, out_dir / "events_out.csv")
    dump_df(alerts_out, out_dir / "alerts_out.csv")
    dump_json(full_summary, out_dir / "summary.json")
    logger.info(f"wrote stratified eval outputs under: {out_dir}")


if __name__ == "__main__":
    main()
