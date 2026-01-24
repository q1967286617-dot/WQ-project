from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass
class ImputeScaleStats:
    median: pd.Series
    mean: pd.Series
    std: pd.Series


def fit_train_imputer_and_scaler(train_df: pd.DataFrame, num_cols: List[str]) -> ImputeScaleStats:
    """
    Fit imputation & scaling stats on TRAIN ONLY.

    Default:
      - impute numeric columns with median
      - scaling is optional (off by default in apply_*)
    """
    med = train_df[num_cols].median(numeric_only=True)
    mu = train_df[num_cols].mean(numeric_only=True)
    sd = train_df[num_cols].std(numeric_only=True).replace(0, 1.0)
    return ImputeScaleStats(median=med, mean=mu, std=sd)


def apply_imputer_and_scaler(
    df: pd.DataFrame,
    num_cols: List[str],
    stats: ImputeScaleStats,
    do_scale: bool = True,
) -> pd.DataFrame:
    x = df.copy()
    # Add missing numeric columns if not present (common when evaluating on different feature sets)
    for c in num_cols:
        if c not in x.columns:
            x[c] = np.nan
    x[num_cols] = x[num_cols].fillna(stats.median)

    if do_scale:
        x[num_cols] = (x[num_cols] - stats.mean) / stats.std
    return x


def prepare_categorical(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cat_cols: List[str],
    missing_token: str = "<<MISSING>>",
    unk_token: str = "<<UNK>>",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Ensure categorical columns are pandas.Categorical with categories defined by TRAIN.

    - Missing values -> <<MISSING>>
    - Unseen categories in val/test -> <<UNK>>
    """
    train = train_df.copy()
    val = val_df.copy()
    test = test_df.copy()

    for c in cat_cols:
        for x in (train, val, test):
            if c not in x.columns:
                x[c] = missing_token

        train[c] = train[c].astype("string").fillna(missing_token)
        val[c] = val[c].astype("string").fillna(missing_token)
        test[c] = test[c].astype("string").fillna(missing_token)

        base_cats = pd.Index(train[c].unique())
        if unk_token not in base_cats:
            base_cats = base_cats.append(pd.Index([unk_token]))

        train[c] = pd.Categorical(train[c], categories=base_cats)
        val[c] = pd.Categorical(val[c].where(val[c].isin(base_cats), unk_token), categories=base_cats)
        test[c] = pd.Categorical(test[c].where(test[c].isin(base_cats), unk_token), categories=base_cats)

    return train, val, test


def split_xy(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str], target_col: str) -> Tuple[pd.DataFrame, np.ndarray]:
    X = df[num_cols + cat_cols].copy()
    y = df[target_col].astype(int).values
    return X, y
