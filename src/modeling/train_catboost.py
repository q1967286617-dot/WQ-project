from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from .preprocess import ImputeScaleStats, apply_imputer_and_scaler


@dataclass
class CatBoostArtifacts:
    model: CatBoostClassifier
    impute_stats: ImputeScaleStats
    num_cols: List[str]
    cat_cols: List[str]
    target_col: str
    feature_names: List[str]
    do_scale: bool = False


def _prepare_cat_strings(x: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    """CatBoost takes raw string categories; replace NaN with sentinel."""
    x = x.copy()
    for c in cat_cols:
        if c not in x.columns:
            x[c] = "<<MISSING>>"
        x[c] = x[c].astype(str).replace("nan", "<<MISSING>>").fillna("<<MISSING>>")
    return x


def train_catboost_binary(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_cols: List[str],
    pos: int,
    neg: int,
    seed: int = 42,
    depth: int = 6,
    learning_rate: float = 0.05,
    iterations: int = 500,
    early_stopping_rounds: int = 20,
    l2_leaf_reg: float = 3.0,
    verbose: int = 50,
) -> CatBoostClassifier:
    scale_pos_weight = float(neg / max(pos, 1))
    cols = X_train.columns.tolist()
    cat_feature_indices = [cols.index(c) for c in cat_cols if c in cols]

    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        loss_function="Logloss",
        eval_metric="AUC",
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=early_stopping_rounds,
        cat_features=cat_feature_indices,
        random_seed=seed,
        verbose=verbose,
        use_best_model=True,
    )

    train_pool = Pool(X_train, label=y_train, cat_features=cat_feature_indices)
    val_pool = Pool(X_val, label=y_val, cat_features=cat_feature_indices)
    model.fit(train_pool, eval_set=val_pool)
    return model


def predict_to_eval_df(
    df: pd.DataFrame,
    art: CatBoostArtifacts,
    permno_col: str = "PERMNO",
    date_col: str = "DlyCalDt",
    keep_extra_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    x = df.copy()
    x[date_col] = pd.to_datetime(x[date_col], errors="coerce")
    x[permno_col] = x[permno_col].astype(int)

    x = apply_imputer_and_scaler(x, art.num_cols, art.impute_stats, do_scale=art.do_scale)
    x = _prepare_cat_strings(x, art.cat_cols)

    X = x[art.feature_names].copy()
    cols = X.columns.tolist()
    cat_feature_indices = [cols.index(c) for c in art.cat_cols if c in cols]
    pool = Pool(X, cat_features=cat_feature_indices)
    prob = art.model.predict_proba(pool)[:, 1]

    y = x[art.target_col].astype(int).values if art.target_col in x.columns else np.full(len(x), -1, dtype=int)
    eval_df = pd.DataFrame({
        "date": x[date_col].values,
        "permno": x[permno_col].values,
        "y": y,
        "prob": prob.astype(float),
    })
    if keep_extra_cols:
        for c in keep_extra_cols:
            if c in x.columns and c not in eval_df.columns:
                eval_df[c] = x[c].values
    return eval_df.sort_values(["permno", "date"]).reset_index(drop=True)


def save_artifacts(art: CatBoostArtifacts, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(art, path)


def load_artifacts(path: str | Path) -> CatBoostArtifacts:
    return joblib.load(path)
