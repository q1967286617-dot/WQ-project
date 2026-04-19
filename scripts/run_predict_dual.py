"""
Dual-booster prediction with cross-sectional rank fusion.

Loads the classifier artifact (xgb_<run_id>.joblib) and the regressor
artifact (xgb_reg_<run_id>.joblib) trained by run_train_dual.py, scores
the split, rank-normalizes both scores **within each trading day**, and
writes a single preds parquet where the `prob` column is the fused score:

    prob = alpha * rank(score_cls) + (1 - alpha) * rank(score_reg)

Downstream run_eval.py / run_backtest.py consume the `prob` column
unchanged. Raw scores are preserved as `score_cls` and `score_reg` for
diagnostics.
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import load_yaml, resolve_paths, ensure_dir
from src.utils.logging import get_logger
from src.modeling.train import load_artifacts, build_dmatrix
from src.modeling.preprocess import apply_imputer_and_scaler


logger = get_logger("run_predict_dual")


def _read_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _score_with_artifact(
    df: pd.DataFrame,
    art,
    permno_col: str,
    date_col: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Apply an artifact's preprocessing and return (scores, processed_df)."""
    x = df.copy()
    x[date_col] = pd.to_datetime(x[date_col], errors="coerce")
    x[permno_col] = x[permno_col].astype(int)

    x = apply_imputer_and_scaler(x, art.num_cols, art.impute_stats, do_scale=art.do_scale)

    for c in art.cat_cols:
        if c not in x.columns:
            x[c] = "<<MISSING>>"
        x[c] = x[c].astype("string").fillna("<<MISSING>>")
        cats = art.cat_categories.get(c)
        if cats:
            cats_idx = pd.Index(cats)
            unk = "<<UNK>>"
            if unk not in cats_idx:
                cats_idx = cats_idx.append(pd.Index([unk]))
            x[c] = pd.Categorical(x[c].where(x[c].isin(cats_idx), unk), categories=cats_idx)
        else:
            x[c] = x[c].astype("category")

    feat_names = art.feature_names or (art.num_cols + art.cat_cols)
    X = x.copy()
    for c in feat_names:
        if c not in X.columns:
            X[c] = np.nan
    X = X[feat_names]
    dm = build_dmatrix(X, y=None, enable_categorical=True, feature_names=feat_names)
    scores = art.booster.predict(dm).astype(float)
    return scores, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--cls_artifacts", default=None)
    ap.add_argument("--reg_artifacts", default=None)
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Fusion weight on classifier rank. 1.0 = pure cls, 0.0 = pure reg.",
    )
    ap.add_argument(
        "--out_suffix",
        default=None,
        help="Optional suffix for the preds file, e.g. '_alpha05'. "
             "If omitted, overwrites the standard <split>_preds.parquet.",
    )
    args = ap.parse_args()

    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError(f"--alpha must be in [0, 1], got {args.alpha}")

    paths_cfg = load_yaml(Path(args.paths))
    paths = resolve_paths(paths_cfg, project_root=Path.cwd())

    run_dir = paths.outputs_dir / args.run_id
    ensure_dir(run_dir / "preds")

    cls_path = Path(args.cls_artifacts) if args.cls_artifacts else paths.models_dir / f"xgb_{args.run_id}.joblib"
    reg_path = Path(args.reg_artifacts) if args.reg_artifacts else paths.models_dir / f"xgb_reg_{args.run_id}.joblib"
    if not cls_path.exists():
        raise FileNotFoundError(f"Missing classifier artifacts: {cls_path}")
    if not reg_path.exists():
        raise FileNotFoundError(f"Missing regressor artifacts: {reg_path}")

    split_path = paths.processed_dir / f"{args.split}.parquet"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    df = _read_df(split_path)

    art_cls = load_artifacts(cls_path)
    art_reg = load_artifacts(reg_path)

    permno_col, date_col = "PERMNO", "DlyCalDt"

    logger.info(f"scoring classifier: {cls_path.name}")
    score_cls, x_cls = _score_with_artifact(df, art_cls, permno_col, date_col)
    logger.info(f"scoring regressor:  {reg_path.name}")
    score_reg, _ = _score_with_artifact(df, art_reg, permno_col, date_col)

    # x_cls is already preprocessed + categorical-aligned; we re-use its index/columns.
    y = x_cls[art_cls.target_col].astype(int).values if art_cls.target_col in x_cls.columns else np.full(len(x_cls), -1, dtype=int)

    eval_df = pd.DataFrame({
        "date": x_cls[date_col].values,
        "permno": x_cls[permno_col].values,
        "y": y,
        "score_cls": score_cls,
        "score_reg": score_reg,
    })

    # Cross-sectional rank normalization per trading day (pct rank in [0, 1])
    eval_df["rank_cls"] = eval_df.groupby("date")["score_cls"].rank(method="average", pct=True)
    eval_df["rank_reg"] = eval_df.groupby("date")["score_reg"].rank(method="average", pct=True)
    eval_df["prob"] = args.alpha * eval_df["rank_cls"] + (1.0 - args.alpha) * eval_df["rank_reg"]

    # Cohort columns used by downstream diagnostics
    keep = ["log_mkt_cap", "turnover_5d", "vol_21d", "SICCD", "industry", "has_div_history"]
    for c in keep:
        if c in x_cls.columns and c not in eval_df.columns:
            eval_df[c] = x_cls[c].values

    # Daily spearman between cls and reg scores — a quick signal-independence check
    per_day_corr = (
        eval_df.groupby("date")[["rank_cls", "rank_reg"]]
        .corr()
        .unstack()
        .iloc[:, 1]
    )
    logger.info(
        "cross-sectional rank corr(cls, reg): median=%.3f, mean=%.3f, p10=%.3f, p90=%.3f",
        float(per_day_corr.median()), float(per_day_corr.mean()),
        float(per_day_corr.quantile(0.1)), float(per_day_corr.quantile(0.9)),
    )
    logger.info(
        "alpha=%.2f -> fused prob: mean=%.3f, std=%.3f",
        args.alpha, float(eval_df["prob"].mean()), float(eval_df["prob"].std()),
    )

    eval_df = eval_df.sort_values(["permno", "date"]).reset_index(drop=True)

    suffix = args.out_suffix or ""
    out_path = run_dir / "preds" / f"{args.split}_preds{suffix}.parquet"
    try:
        eval_df.to_parquet(out_path, index=False)
        logger.info(f"saved: {out_path}")
    except Exception as e:
        out_csv = out_path.with_suffix(".csv")
        eval_df.to_csv(out_csv, index=False)
        logger.info(f"parquet unavailable ({e}); saved CSV instead: {out_csv}")


if __name__ == "__main__":
    main()
