from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from .preprocess import ImputeScaleStats, apply_imputer_and_scaler


@dataclass
class LRArtifacts:
    model: LogisticRegression
    ohe: OneHotEncoder
    impute_stats: ImputeScaleStats
    num_cols: List[str]
    cat_cols: List[str]
    target_col: str
    do_scale: bool = True  # LR requires scaling


def _build_X(x: pd.DataFrame, num_cols: List[str], cat_cols: List[str], ohe: OneHotEncoder) -> np.ndarray:
    X_num = x[num_cols].values
    cat_data = x[cat_cols].astype(str).fillna("<<MISSING>>")
    X_cat = ohe.transform(cat_data)
    return np.hstack([X_num, X_cat])


def train_lr_binary(
    X_train_num: np.ndarray,
    X_train_cat: np.ndarray,
    y_train: np.ndarray,
    C: float = 0.1,
    penalty: str = "l1",
    solver: str = "saga",
    max_iter: int = 1000,
    seed: int = 42,
) -> LogisticRegression:
    X = np.hstack([X_train_num, X_train_cat])
    model = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        class_weight="balanced",
        max_iter=max_iter,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X, y_train)
    return model


def predict_to_eval_df(
    df: pd.DataFrame,
    art: LRArtifacts,
    permno_col: str = "PERMNO",
    date_col: str = "DlyCalDt",
    keep_extra_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    x = df.copy()
    x[date_col] = pd.to_datetime(x[date_col], errors="coerce")
    x[permno_col] = x[permno_col].astype(int)

    x = apply_imputer_and_scaler(x, art.num_cols, art.impute_stats, do_scale=art.do_scale)
    X = _build_X(x, art.num_cols, art.cat_cols, art.ohe)
    prob = art.model.predict_proba(X)[:, 1]

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


def save_artifacts(art: LRArtifacts, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(art, path)


def load_artifacts(path: str | Path) -> LRArtifacts:
    return joblib.load(path)
