from __future__ import annotations

from typing import Dict, Tuple
import yaml

import pandas as pd

from . import CONFIG_DIR, DATA_DIR
from .load import PERMNO_COL, DATE_COL

def split_by_date(df: pd.DataFrame, split_days: Dict):

    TRAIN_START = split_days['train_start']
    TRAIN_END   = split_days['train_end']
    VAL_START   = split_days['val_start']
    VAL_END     = split_days['val_end']
    TEST_START  = split_days['test_start']
    TEST_END    = split_days['test_end']

    """按时间切分（这里不 purge，purge 另做）"""
    df = df.copy()
    d = df[DATE_COL]

    train = df[(d >= TRAIN_START) & (d <= TRAIN_END)]
    val   = df[(d >= VAL_START)   & (d <= VAL_END)]
    test  = df[(d >= TEST_START)  & (d <= TEST_END)]
    return train, val, test


def purge_tail(df: pd.DataFrame, end_date: str, embargo_td: int):
    """
    时间序列常用 purge/embargo：把 end_date 之前最后 embargo_td 个交易日样本去掉
    这里用“按日期排序后的最后 N 行”近似交易日（你的 df 本身就是交易日序列）
    """
    x = df.sort_values([PERMNO_COL, DATE_COL]).copy()
    x = x[x[DATE_COL] <= pd.to_datetime(end_date)]
    # 对每个股票都 purge 最后 embargo_td 行（更严格）
    x["_rank_desc"] = x.groupby(PERMNO_COL).cumcount(ascending=False)
    x = x[x["_rank_desc"] >= embargo_td].drop(columns=["_rank_desc"])
    return x

def main():
    with open(CONFIG_DIR / "config.yaml", "r", encoding="utf-8") as f:
        # 使用 safe_load 避免执行任意代码，这是安全实践
        data = yaml.safe_load(f)

    df_full_labeled = pd.read_parquet(DATA_DIR / "interim" / "stage3" / "df_full_labeled.parquet")
    
    train_df, val_df, test_df = split_by_date(df_full_labeled, data['split'])

    output_dir = Path("./data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(output_dir / 'train.parquet', index=False)
    val_df.to_parquet(output_dir / 'val.parquet', index=False)
    test_df.to_parquet(output_dir / 'test.parquet', index=False)
    
    print(f"文件已成功保存至: {output_dir}")
if __name__ == "__main__":
    main()