"""Walk-forward orchestrator.

For each annual test year:
  1. slice the global panel into fold-specific train/val/test,
  2. write them to data/processed/<suffix>_train.parquet etc.,
  3. invoke run_train.py + run_predict.py with --data_suffix <fold_suffix>,
  4. collect per-fold test predictions,
  5. concatenate predictions across folds and write to outputs/runs/<combined>/preds/,
  6. invoke run_backtest.py once on the combined preds.

Per-fold model artifacts land at models/xgb_<combined>__<fold>.joblib.

Example:
  python scripts/run_walk_forward.py \
      --combined_id wf_v12_5y \
      --model_cfg configs/versions/v12_model.yaml \
      --backtest_cfg configs/versions/v12_backtest.yaml \
      --data_suffix _with_fundamentals \
      --test_start 2023 --test_end 2024 --train_years 5
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1]))

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from src.data.walk_forward import build_annual_folds, slice_panel_for_fold
from src.utils.paths import load_yaml, resolve_paths, ensure_dir
from src.utils.logging import get_logger


logger = get_logger("run_walk_forward")
PY = sys.executable


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _load_full_panel(processed_dir: Path, suffix: str) -> pd.DataFrame:
    parts = []
    for name in ["train", "val", "test"]:
        p = processed_dir / f"{name}{suffix}.parquet"
        if not p.exists():
            raise FileNotFoundError(p)
        parts.append(_read_parquet(p))
    df = pd.concat(parts, ignore_index=True)
    df["DlyCalDt"] = pd.to_datetime(df["DlyCalDt"], errors="coerce")
    return df.sort_values(["PERMNO", "DlyCalDt"]).reset_index(drop=True)


def _run_subprocess(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"$ {' '.join(cmd)}")
    with log_path.open("w", encoding="utf-8") as f:
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return res.returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--model_cfg", required=True)
    ap.add_argument("--backtest_cfg", required=True)
    ap.add_argument("--combined_id", required=True, help="run_id for the stitched-across-folds run")
    ap.add_argument("--data_suffix", default="", help="Original split files suffix (e.g. _with_fundamentals)")
    ap.add_argument("--test_start", type=int, default=2023)
    ap.add_argument("--test_end",   type=int, default=2024)
    ap.add_argument("--train_years", type=int, default=5)
    ap.add_argument("--val_years",   type=int, default=1)
    ap.add_argument("--mode", choices=["rolling", "expanding"], default="rolling")
    ap.add_argument("--embargo_days", type=int, default=14)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep_fold_parquets", action="store_true",
                    help="Keep per-fold processed parquets after each fold finishes")
    ap.add_argument("--warmup_days", type=int, default=0,
                    help="Calendar days before first fold's test_start to also predict, "
                         "using the first fold's model. Lets simulate_portfolio plan "
                         "carry-in entries that mirror what single-train backtest sees, "
                         "removing first-fold boundary bias. Recommend ~30 (>= holding_td days).")
    ap.add_argument("--capture_val_preds", action="store_true",
                    help="Also run predict on each fold's val split and concatenate to "
                         "outputs/<combined_id>/preds/val_preds.parquet. Needed for "
                         "per-fold parameter sweep (run_wf_param_sweep.py).")
    args = ap.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    paths = resolve_paths(paths_cfg, project_root=Path.cwd())
    processed_dir = paths.processed_dir
    models_dir    = paths.models_dir
    outputs_dir   = paths.outputs_dir
    ensure_dir(models_dir); ensure_dir(outputs_dir)

    log_dir = outputs_dir / args.combined_id / "walk_forward_logs"
    ensure_dir(log_dir)

    logger.info("loading full panel (train+val+test concat)")
    full = _load_full_panel(processed_dir, args.data_suffix)
    logger.info(f"full panel: rows={len(full)} dates={full['DlyCalDt'].min().date()}..{full['DlyCalDt'].max().date()}")

    folds = build_annual_folds(
        test_start_year=args.test_start,
        test_end_year=args.test_end,
        train_years=args.train_years,
        val_years=args.val_years,
        mode=args.mode,
        embargo_calendar_days=args.embargo_days,
    )
    logger.info(f"generated {len(folds)} folds")
    for f in folds:
        logger.info(f"  {f.fold_id}: train {f.train_start}..{f.train_end} | val {f.val_start}..{f.val_end} | test {f.test_start}..{f.test_end}")

    fold_summaries: list[dict] = []
    test_preds_parts: list[pd.DataFrame] = []
    val_preds_parts:  list[pd.DataFrame] = []

    for fold in folds:
        logger.info(f"=== fold {fold.fold_id} ===")
        tr, va, te = slice_panel_for_fold(full, fold)
        logger.info(f"  rows train={len(tr)} val={len(va)} test={len(te)}")
        if tr.empty or va.empty or te.empty:
            logger.warning(f"  skipping {fold.fold_id}: empty slice")
            continue

        fold_suffix = f"_{args.combined_id}__{fold.fold_id}"
        tr_path = processed_dir / f"train{fold_suffix}.parquet"
        va_path = processed_dir / f"val{fold_suffix}.parquet"
        te_path = processed_dir / f"test{fold_suffix}.parquet"
        tr.to_parquet(tr_path, index=False)
        va.to_parquet(va_path, index=False)
        te.to_parquet(te_path, index=False)

        fold_run_id = f"{args.combined_id}__{fold.fold_id}"

        rc = _run_subprocess([PY, "scripts/run_train.py",
                              "--paths", args.paths,
                              "--model_cfg", args.model_cfg,
                              "--run_id", fold_run_id,
                              "--data_suffix", fold_suffix,
                              "--seed", str(args.seed)],
                             log_dir / f"{fold.fold_id}_train.log")
        if rc != 0:
            logger.error(f"  train failed (rc={rc}); see {log_dir / f'{fold.fold_id}_train.log'}")
            raise SystemExit(rc)

        rc = _run_subprocess([PY, "scripts/run_predict.py",
                              "--paths", args.paths,
                              "--run_id", fold_run_id,
                              "--split", "test",
                              "--data_suffix", fold_suffix],
                             log_dir / f"{fold.fold_id}_predict.log")
        if rc != 0:
            logger.error(f"  predict failed (rc={rc})")
            raise SystemExit(rc)

        if args.capture_val_preds:
            rc = _run_subprocess([PY, "scripts/run_predict.py",
                                  "--paths", args.paths,
                                  "--run_id", fold_run_id,
                                  "--split", "val",
                                  "--data_suffix", fold_suffix],
                                 log_dir / f"{fold.fold_id}_predict_val.log")
            if rc != 0:
                logger.warning(f"  val predict failed (rc={rc}); skipping val preds for {fold.fold_id}")
            else:
                val_preds_path = outputs_dir / fold_run_id / "preds" / "val_preds.parquet"
                if val_preds_path.exists():
                    vp = pd.read_parquet(val_preds_path)
                    vp["date"] = pd.to_datetime(vp["date"], errors="coerce")
                    vmask = (vp["date"] >= pd.to_datetime(fold.val_start)) & (vp["date"] <= pd.to_datetime(fold.val_end))
                    val_preds_parts.append(vp[vmask].copy())
                    logger.info(f"  val preds captured: {vmask.sum()} rows")

        preds_path = outputs_dir / fold_run_id / "preds" / "test_preds.parquet"
        if not preds_path.exists():
            raise FileNotFoundError(preds_path)
        preds = pd.read_parquet(preds_path)
        # Defensive: keep only this fold's test window
        preds["date"] = pd.to_datetime(preds["date"], errors="coerce")
        mask = (preds["date"] >= pd.to_datetime(fold.test_start)) & (preds["date"] <= pd.to_datetime(fold.test_end))
        preds = preds[mask].copy()
        test_preds_parts.append(preds)
        fold_summaries.append({
            "fold_id":        fold.fold_id,
            "train":          [fold.train_start, fold.train_end, len(tr)],
            "val":            [fold.val_start,   fold.val_end,   len(va)],
            "test":           [fold.test_start,  fold.test_end,  len(te)],
            "pred_rows":      int(len(preds)),
        })

        if not args.keep_fold_parquets:
            tr_path.unlink(missing_ok=True)
            va_path.unlink(missing_ok=True)
            te_path.unlink(missing_ok=True)

    if not test_preds_parts:
        raise SystemExit("no folds produced predictions")

    # Warmup: predict on the N calendar days immediately before the first
    # fold's test_start using the FIRST fold's model. Without this, the
    # combined backtest cannot plan entries on the first ~holding_td days
    # of test_start because the signals that would have triggered them
    # (from the prior trading days) are missing from the panel. Single-
    # train backtest has those signals naturally because its panel begins
    # at the test split's first day; WF's panel begins at test_start.
    if args.warmup_days > 0 and folds:
        first = folds[0]
        warmup_start = (pd.to_datetime(first.test_start) - pd.Timedelta(days=int(args.warmup_days))).strftime("%Y-%m-%d")
        warmup_end   = (pd.to_datetime(first.test_start) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        d_full = pd.to_datetime(full["DlyCalDt"], errors="coerce")
        warmup_slice = full[(d_full >= pd.to_datetime(warmup_start)) & (d_full <= pd.to_datetime(warmup_end))].copy()
        if not warmup_slice.empty:
            from src.modeling.predict import predict_to_eval_df_dispatch
            first_run_id = f"{args.combined_id}__{first.fold_id}"
            # Auto-locate first fold's artifact (run_predict's lookup logic)
            art_path = None
            for prefix in ["xgb", "lgbm", "catboost", "lr"]:
                cand = models_dir / f"{prefix}_{first_run_id}.joblib"
                if cand.exists():
                    art_path = cand
                    break
            if art_path is None:
                raise FileNotFoundError(f"no artifact for first fold {first_run_id}")
            logger.info(f"warmup predict: {warmup_start}..{warmup_end} ({len(warmup_slice)} rows) using {art_path.name}")
            keep = ["log_mkt_cap", "turnover_5d", "vol_21d", "SICCD", "industry", "has_div_history"]
            warmup_preds = predict_to_eval_df_dispatch(warmup_slice, art_path, keep_extra_cols=keep)
            warmup_preds["date"] = pd.to_datetime(warmup_preds["date"], errors="coerce")
            wmask = (warmup_preds["date"] >= pd.to_datetime(warmup_start)) & (warmup_preds["date"] <= pd.to_datetime(warmup_end))
            warmup_preds = warmup_preds[wmask].copy()
            logger.info(f"warmup preds: {len(warmup_preds)} rows from first fold's model")
            test_preds_parts.insert(0, warmup_preds)

    combined_preds = pd.concat(test_preds_parts, ignore_index=True)
    combined_preds = combined_preds.drop_duplicates(subset=["date", "permno"], keep="last").sort_values(["permno", "date"]).reset_index(drop=True)
    combined_run_dir = outputs_dir / args.combined_id
    ensure_dir(combined_run_dir / "preds")
    combined_preds_path = combined_run_dir / "preds" / "test_preds.parquet"
    combined_preds.to_parquet(combined_preds_path, index=False)
    logger.info(f"wrote combined preds: {combined_preds_path}  rows={len(combined_preds)}")

    if val_preds_parts:
        combined_val = pd.concat(val_preds_parts, ignore_index=True)
        combined_val = combined_val.drop_duplicates(subset=["date", "permno"], keep="last").sort_values(["permno", "date"]).reset_index(drop=True)
        combined_val_path = combined_run_dir / "preds" / "val_preds.parquet"
        combined_val.to_parquet(combined_val_path, index=False)
        logger.info(f"wrote combined val preds: {combined_val_path}  rows={len(combined_val)}")

    (combined_run_dir / "folds.json").write_text(
        json.dumps({"folds": fold_summaries}, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Single backtest on the stitched predictions.
    # The original test.parquet (covering the union of fold test windows) is
    # used by run_backtest to fetch execution prices etc.
    rc = _run_subprocess([PY, "scripts/run_backtest.py",
                          "--paths", args.paths,
                          "--run_id", args.combined_id,
                          "--split", "test",
                          "--backtest_cfg", args.backtest_cfg,
                          "--skip_reference"],
                         log_dir / "combined_backtest.log")
    if rc != 0:
        logger.error(f"backtest failed (rc={rc})")
        raise SystemExit(rc)

    summary_path = combined_run_dir / "backtest" / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        port = summary.get("portfolio", {})
        logger.info(f"WALK-FORWARD COMBINED: sharpe={port.get('sharpe'):.4f} "
                    f"ann_ret={port.get('annualized_return')*100:.2f}% "
                    f"max_dd={port.get('max_drawdown')*100:.2f}%")
    else:
        logger.warning("no backtest summary produced")


if __name__ == "__main__":
    main()
