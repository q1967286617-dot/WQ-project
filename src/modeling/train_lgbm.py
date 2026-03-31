from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from .preprocess import ImputeScaleStats, apply_imputer_and_scaler


@dataclass
class LGBMArtifacts:
    booster: lgb.Booster
    impute_stats: ImputeScaleStats
    num_cols: List[str]
    cat_cols: List[str]
    cat_categories: Dict[str, List[str]]
    target_col: str
    feature_names: List[str]
    do_scale: bool = False


def _apply_cat_categories(x: pd.DataFrame, cat_cols: List[str], cat_categories: Dict[str, List[str]]) -> pd.DataFrame:
    """Reapply train-time category mappings (mirrors XGBoost predict.py logic)."""
    x = x.copy()
    for c in cat_cols:
        if c not in x.columns:
            x[c] = "<<MISSING>>"
        x[c] = x[c].astype("string").fillna("<<MISSING>>")
        cats = cat_categories.get(c)
        if cats:
            cats_idx = pd.Index(cats)
            unk = "<<UNK>>"
            if unk not in cats_idx:
                cats_idx = cats_idx.append(pd.Index([unk]))
            x[c] = pd.Categorical(x[c].where(x[c].isin(cats_idx), unk), categories=cats_idx)
        else:
            x[c] = x[c].astype("category")
    return x


def train_lgbm_binary(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_cols: List[str],
    pos: int,
    neg: int,
    seed: int = 42,
    num_leaves: int = 63,
    learning_rate: float = 0.05,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 20,
    subsample: float = 0.8,
    feature_fraction: float = 0.8,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
    min_child_samples: int = 20,
    verbose_eval: int = 50,
) -> lgb.Booster:
    scale_pos_weight = float(neg / max(pos, 1))

    params: Dict[str, Any] = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc", "average_precision"],
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "bagging_fraction": subsample,
        "bagging_freq": 1,
        "feature_fraction": feature_fraction,
        "lambda_l2": reg_lambda,
        "lambda_l1": reg_alpha,
        "min_child_samples": min_child_samples,
        "scale_pos_weight": scale_pos_weight,
        "seed": seed,
        "verbosity": -1,
        "n_jobs": -1,
    }

    cat_feature = [c for c in cat_cols if c in X_train.columns]
    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_feature, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_feature, reference=dtrain, free_raw_data=False)

    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
        lgb.log_evaluation(period=verbose_eval),
    ]

    booster = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )
    return booster


def predict_to_eval_df(
    df: pd.DataFrame,
    art: LGBMArtifacts,
    permno_col: str = "PERMNO",
    date_col: str = "DlyCalDt",
    keep_extra_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    x = df.copy()
    x[date_col] = pd.to_datetime(x[date_col], errors="coerce")
    x[permno_col] = x[permno_col].astype(int)

    x = apply_imputer_and_scaler(x, art.num_cols, art.impute_stats, do_scale=art.do_scale)
    x = _apply_cat_categories(x, art.cat_cols, art.cat_categories)

    X = x[art.feature_names].copy()
    prob = art.booster.predict(X, num_iteration=art.booster.best_iteration)

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


def save_artifacts(art: LGBMArtifacts, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(art, path)


def load_artifacts(path: str | Path) -> LGBMArtifacts:
    return joblib.load(path)
