from __future__ import annotations

from typing import Dict

import pandas as pd
import yaml

from . import CONFIG_DIR, DATA_DIR
from .load import DATE_COL, PERMNO_COL


def split_by_date(df: pd.DataFrame, split_days: Dict):
    train_start = split_days["train_start"]
    train_end = split_days["train_end"]
    val_start = split_days["val_start"]
    val_end = split_days["val_end"]
    test_start = split_days["test_start"]
    test_end = split_days["test_end"]

    d = df[DATE_COL]
    train = df[(d >= train_start) & (d <= train_end)]
    val = df[(d >= val_start) & (d <= val_end)]
    test = df[(d >= test_start) & (d <= test_end)]
    return train, val, test


def purge_tail(df: pd.DataFrame, end_date: str, embargo_td: int):
    x = df.sort_values([PERMNO_COL, DATE_COL]).copy()
    x = x[x[DATE_COL] <= pd.to_datetime(end_date)]
    x["_rank_desc"] = x.groupby(PERMNO_COL).cumcount(ascending=False)
    x = x[x["_rank_desc"] >= embargo_td].drop(columns=["_rank_desc"])
    return x


def main():
    with open(CONFIG_DIR / "config.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    df_full_labeled = pd.read_parquet(DATA_DIR / "interim" / "stage3" / "df_full_labeled.parquet")
    train_df, val_df, test_df = split_by_date(df_full_labeled, data["split"])

    output_dir = DATA_DIR / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "val.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)

    print(f"Data Splitted: {output_dir}")


if __name__ == "__main__":
    main()
