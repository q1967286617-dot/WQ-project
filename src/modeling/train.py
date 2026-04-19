from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Any, Dict

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from .preprocess import ImputeScaleStats


@dataclass
class TrainArtifacts:
    """
    Bundles everything needed to reproduce predictions:
      - booster: trained xgboost Booster
      - impute_stats: numeric imputation/scaling stats
      - num_cols/cat_cols: feature schema
      - target_col: label column name
      - feature_names: final feature order used in training
      - do_scale: whether numeric scaling was used
    """
    booster: xgb.Booster
    impute_stats: ImputeScaleStats
    num_cols: List[str]
    cat_cols: List[str]
    cat_categories: Dict[str, List[str]]
    target_col: str
    feature_names: List[str]
    do_scale: bool = False


def _auto_device() -> str:
    # xgboost uses CUDA only if built with GPU and a GPU is available; safe fallback to cpu
    try:
        import cupy  # noqa: F401
        return "cuda"
    except Exception:
        return "cpu"


def build_dmatrix(
    X: pd.DataFrame,
    y: Optional[np.ndarray] = None,
    enable_categorical: bool = True,
    feature_names: Optional[List[str]] = None,
) -> xgb.DMatrix:
    """
    Build an XGBoost DMatrix. If feature_names is provided, columns will be reordered/expanded.
    """
    if feature_names is not None:
        X = X.copy()
        for c in feature_names:
            if c not in X.columns:
                X[c] = np.nan
        X = X[feature_names]
    return xgb.DMatrix(
        X,
        label=y,
        enable_categorical=enable_categorical,
        missing=np.nan,
        feature_names=feature_names,
    )


def train_xgb_binary(
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
    pos: int,
    neg: int,
    seed: int = 42,
    nthread: int = -1,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 20,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    reg_alpha: float = 1.0,
    min_child_weight: float = 1.0,
    device: Optional[str] = None,
    verbose_eval: int = 50,
) -> xgb.Booster:
    """
    Train an XGBoost binary classifier with early stopping.
    Supports numeric + categorical features (enable_categorical in DMatrix).
    """
    if device is None:
        device = _auto_device()

    # Avoid division by zero
    spw = float(neg / max(pos, 1))

    params: Dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc", "aucpr"],
        "tree_method": "hist",
        "device": device,

        "max_depth": int(max_depth),
        "min_child_weight": float(min_child_weight),
        "subsample": float(subsample),
        "colsample_bytree": float(colsample_bytree),
        "learning_rate": float(learning_rate),
        "reg_lambda": float(reg_lambda),
        "reg_alpha": float(reg_alpha),

        "scale_pos_weight": spw,
        "seed": int(seed),
        "nthread": int(nthread),
    }

    watchlist = [(dtrain, "train"), (dval, "val")]

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=int(num_boost_round),
        evals=watchlist,
        early_stopping_rounds=int(early_stopping_rounds),
        verbose_eval=verbose_eval,
    )
    return booster


@dataclass
class RegArtifacts:
    """Same shape as TrainArtifacts, but for a regression booster."""
    booster: xgb.Booster
    impute_stats: ImputeScaleStats
    num_cols: List[str]
    cat_cols: List[str]
    cat_categories: Dict[str, List[str]]
    target_col: str
    feature_names: List[str]
    do_scale: bool = False


def train_xgb_regressor(
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
    seed: int = 42,
    nthread: int = -1,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 20,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    reg_alpha: float = 1.0,
    min_child_weight: float = 1.0,
    objective: str = "reg:squarederror",
    device: Optional[str] = None,
    verbose_eval: int = 50,
) -> xgb.Booster:
    """Train an XGBoost regressor with early stopping on RMSE."""
    if device is None:
        device = _auto_device()

    params: Dict[str, Any] = {
        "objective": objective,
        "eval_metric": ["rmse"],
        "tree_method": "hist",
        "device": device,

        "max_depth": int(max_depth),
        "min_child_weight": float(min_child_weight),
        "subsample": float(subsample),
        "colsample_bytree": float(colsample_bytree),
        "learning_rate": float(learning_rate),
        "reg_lambda": float(reg_lambda),
        "reg_alpha": float(reg_alpha),

        "seed": int(seed),
        "nthread": int(nthread),
    }

    watchlist = [(dtrain, "train"), (dval, "val")]
    return xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=int(num_boost_round),
        evals=watchlist,
        early_stopping_rounds=int(early_stopping_rounds),
        verbose_eval=verbose_eval,
    )


def train_xgb_ranker(
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
    seed: int = 42,
    nthread: int = -1,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 20,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    reg_alpha: float = 1.0,
    min_child_weight: float = 1.0,
    objective: str = "rank:ndcg",
    lambdarank_pair_method: str = "topk",
    lambdarank_num_pair_per_sample: int = 8,
    device: Optional[str] = None,
    verbose_eval: int = 50,
) -> xgb.Booster:
    """
    Train an XGBoost learning-to-rank booster. Groups must already be set on
    both dtrain and dval (each group == one trading day). Labels should be
    non-negative relevance scores (e.g. per-day quantile bins).
    """
    if device is None:
        device = _auto_device()

    params: Dict[str, Any] = {
        "objective": objective,
        "eval_metric": ["ndcg@20", "ndcg@50"],
        "tree_method": "hist",
        "device": device,
        "lambdarank_pair_method": lambdarank_pair_method,
        "lambdarank_num_pair_per_sample": int(lambdarank_num_pair_per_sample),

        "max_depth": int(max_depth),
        "min_child_weight": float(min_child_weight),
        "subsample": float(subsample),
        "colsample_bytree": float(colsample_bytree),
        "learning_rate": float(learning_rate),
        "reg_lambda": float(reg_lambda),
        "reg_alpha": float(reg_alpha),

        "seed": int(seed),
        "nthread": int(nthread),
    }

    watchlist = [(dtrain, "train"), (dval, "val")]
    return xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=int(num_boost_round),
        evals=watchlist,
        early_stopping_rounds=int(early_stopping_rounds),
        verbose_eval=verbose_eval,
    )


def save_artifacts(art: TrainArtifacts, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(art, path)


def load_artifacts(path: str | Path) -> TrainArtifacts:
    return joblib.load(path)