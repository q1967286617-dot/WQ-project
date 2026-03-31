from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import joblib
import numpy as np
import pandas as pd

from .train import load_artifacts, build_dmatrix, TrainArtifacts
from .preprocess import apply_imputer_and_scaler


def _attach_aidx_per_permno(df: pd.DataFrame, permno_col: str, date_col: str) -> pd.DataFrame:
    x = df.sort_values([permno_col, date_col]).copy()
    x["_aidx"] = x.groupby(permno_col).cumcount().astype(int)
    return x


def predict_to_eval_df(
    df: pd.DataFrame,
    art_path: str | Path,
    permno_col: str = "PERMNO",
    date_col: str = "DlyCalDt",
    keep_extra_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Convert a labeled dataset into the canonical eval_df schema:
        date, permno, y, prob, (+ optional extra columns)

    This function:
      - loads TrainArtifacts
      - imputes numeric columns using train stats
      - ensures categorical columns are category dtype with train categories
      - predicts probabilities using the Booster
    """
    art = load_artifacts(art_path)

    x = df.copy()
    x[date_col] = pd.to_datetime(x[date_col], errors="coerce")
    x[permno_col] = x[permno_col].astype(int)

    # Numeric processing (impute; scaling optional)
    x = apply_imputer_and_scaler(x, art.num_cols, art.impute_stats, do_scale=art.do_scale)

    
    # Categorical processing: enforce categories defined by TRAIN (stored in artifacts)
    for c in art.cat_cols:
        if c not in x.columns:
            x[c] = "<<MISSING>>"
        x[c] = x[c].astype("string").fillna("<<MISSING>>")

        cats = art.cat_categories.get(c)
        if cats:
            cats_idx = pd.Index(cats)
            # unseen categories -> <<UNK>> if present, else keep as missing
            unk = "<<UNK>>"
            if unk not in cats_idx:
                cats_idx = cats_idx.append(pd.Index([unk]))
            x[c] = pd.Categorical(x[c].where(x[c].isin(cats_idx), unk), categories=cats_idx)
        else:
            x[c] = x[c].astype("category")
# Assemble X in training order
    # If feature_names exists, enforce exact order; else use num_cols+cat_cols.
    feat_names = art.feature_names or (art.num_cols + art.cat_cols)
    X = x.copy()
    for c in feat_names:
        if c not in X.columns:
            X[c] = np.nan
    X = X[feat_names]

    dm = build_dmatrix(X, y=None, enable_categorical=True, feature_names=feat_names)
    prob = art.booster.predict(dm)

    # Build eval_df
    out_cols = ["date", "permno", "y", "prob"]
    y = x[art.target_col].astype(int).values if art.target_col in x.columns else np.full(len(x), -1, dtype=int)

    eval_df = pd.DataFrame({
        "date": x[date_col].values,
        "permno": x[permno_col].values,
        "y": y,
        "prob": prob.astype(float),
    })

    # Optional: keep extra columns for cohorting / diagnostics
    if keep_extra_cols:
        for c in keep_extra_cols:
            if c in x.columns and c not in eval_df.columns:
                eval_df[c] = x[c].values

    return eval_df.sort_values(["permno", "date"]).reset_index(drop=True)


def predict_to_eval_df_dispatch(
    df: pd.DataFrame,
    art_path: str | Path,
    permno_col: str = "PERMNO",
    date_col: str = "DlyCalDt",
    keep_extra_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Auto-detect artifact type and dispatch to the correct predict function."""
    art = joblib.load(art_path)

    if isinstance(art, TrainArtifacts):
        return predict_to_eval_df(df, art_path, permno_col=permno_col, date_col=date_col, keep_extra_cols=keep_extra_cols)

    from .train_lgbm import LGBMArtifacts
    from .train_lgbm import predict_to_eval_df as _lgbm_pred
    if isinstance(art, LGBMArtifacts):
        return _lgbm_pred(df, art, permno_col=permno_col, date_col=date_col, keep_extra_cols=keep_extra_cols)

    from .train_catboost import CatBoostArtifacts
    from .train_catboost import predict_to_eval_df as _cb_pred
    if isinstance(art, CatBoostArtifacts):
        return _cb_pred(df, art, permno_col=permno_col, date_col=date_col, keep_extra_cols=keep_extra_cols)

    from .train_lr import LRArtifacts
    from .train_lr import predict_to_eval_df as _lr_pred
    if isinstance(art, LRArtifacts):
        return _lr_pred(df, art, permno_col=permno_col, date_col=date_col, keep_extra_cols=keep_extra_cols)

    raise TypeError(f"Unknown artifact type: {type(art)}")
