from __future__ import annotations

from pathlib import Path
from typing import Set, Optional, List

import pandas as pd

import yaml

import joblib

PERMNO_COL = "PERMNO"
DATE_COL = "DlyCalDt"


def build_fixed_universe(table_b_path: Path, start_all: str, end_all: str, chunksize=500_000):
    """扫描 tableB 构建固定 PERMNO 宇宙（只用日期过滤，不做特征、不做切分）"""
    permno_set = set()
    reader = pd.read_csv(table_b_path, usecols=[PERMNO_COL, DATE_COL], chunksize=chunksize)
    for chunk in reader:
        chunk[DATE_COL] = pd.to_datetime(chunk[DATE_COL], errors="coerce")
        chunk = chunk.dropna(subset=[PERMNO_COL, DATE_COL])
        mask = (chunk[DATE_COL] >= start_all) & (chunk[DATE_COL] <= end_all)
        permno_set.update(chunk.loc[mask, PERMNO_COL].astype(int).unique())
    return permno_set


def load_market_data_full(table_b_path: Path, permno_set: set, start_all: str, end_all: str, chunksize=500_000):
    """读出全量行情（固定宇宙 + 日期范围），只做最小清洗与排序"""
    usecols = [PERMNO_COL, "Ticker", DATE_COL, "SICCD", "DlyClose", "DlyPrc", "DlyVol", "DlyRet", "ShrOut"]
    out = []
    reader = pd.read_csv(table_b_path, usecols=usecols, chunksize=chunksize, low_memory=True)

    for chunk in reader:
        chunk[DATE_COL] = pd.to_datetime(chunk[DATE_COL], errors="coerce")
        chunk = chunk.dropna(subset=[PERMNO_COL, DATE_COL])
        chunk[PERMNO_COL] = chunk[PERMNO_COL].astype(int)

        mask = (
            chunk[PERMNO_COL].isin(permno_set)
            & (chunk[DATE_COL] >= start_all) & (chunk[DATE_COL] <= end_all)
        )
        out.append(chunk.loc[mask])

    df = pd.concat(out, ignore_index=True)

    # 强制唯一性（避免 rolling 混乱）
    df = df.drop_duplicates(subset=[PERMNO_COL, DATE_COL])

    # 关键：按 (permno, date) 排序，保证 rolling / shift 正确
    df = df.sort_values([PERMNO_COL, DATE_COL]).reset_index(drop=True)
    return df


def load_div_events(
    table_a_path: Path,
    DIV_DISTCD: list,
    permno_col: str = PERMNO_COL,
    event_date_col: str = "DCLRDT"
) -> pd.DataFrame:
    """
    Load dividend declaration events from tableA (default column: DCLRDT),
    then standardize dtype and sort.
    """
    ev = pd.read_csv(table_a_path, usecols=[permno_col, event_date_col, "DISTCD"])
    ev[permno_col] = ev[permno_col].astype(int)
    ev[event_date_col] = pd.to_datetime(ev[event_date_col], errors="coerce")

    ev = ev.dropna(subset=[permno_col, event_date_col]).drop_duplicates()
    ev = ev.sort_values([permno_col, event_date_col]).reset_index(drop=True)
    ev = ev[ev['DISTCD'].isin(DIV_DISTCD)]
    return ev.drop(columns=['DISTCD'])

def standardize_events(
    events_df: pd.DataFrame,
    permno_col: str = PERMNO_COL,
    event_date_col: str = "DCLRDT",
) -> pd.DataFrame:
    """
    Convert an events table into a canonical schema for evaluation:
        permno: int
        event_date: datetime64[ns]
    """
    ev = events_df.rename(columns={permno_col: "permno", event_date_col: "event_date"}).copy()
    ev["permno"] = ev["permno"].astype(int)
    ev["event_date"] = pd.to_datetime(ev["event_date"], errors="coerce")
    ev = ev.dropna(subset=["permno", "event_date"]).drop_duplicates()
    ev = ev.sort_values(["permno", "event_date"]).reset_index(drop=True)
    return ev

def main():
    with open('./configs/config.yaml', 'r', encoding='utf-8') as f:
        # 使用 safe_load 避免执行任意代码，这是安全实践
        data = yaml.safe_load(f)
    
    TABLE_A_PATH = "./" + data['tableA_path']
    TABLE_B_PATH = "./" + data['tableB_path']

    START_ALL = data['start_all']
    END_ALL = data['end_all']

    DIV_DISTCD = data['div_distcd']

    permno_set = build_fixed_universe(TABLE_B_PATH, START_ALL, END_ALL)
    df_full_raw = load_market_data_full(TABLE_B_PATH, permno_set, START_ALL, END_ALL)

    div_ev = load_div_events(TABLE_A_PATH, DIV_DISTCD)

    output_dir = Path("./data/interim/stage1")
    output_dir.mkdir(parents=True, exist_ok=True)

    df_full_raw.to_parquet(output_dir / 'df_full_raw.parquet', index=False)
    div_ev.to_parquet(output_dir / 'div_ev.parquet', index=False)

    print(f"文件已成功保存至: {output_dir}")

if __name__ == "__main__":
    main()