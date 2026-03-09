from __future__ import annotations

from typing import Dict, Any

import yaml

import numpy as np
import pandas as pd

from . import CONFIG_DIR, DATA_DIR
from .load import PERMNO_COL, DATE_COL


def build_event_dict(events: pd.DataFrame, date_col: str):
    """把事件表变成 {permno: np.array(sorted_dates)}，供打标快速查找"""
    out = {}
    for p, g in events.groupby(PERMNO_COL):
        out[int(p)] = np.sort(g[date_col].values.astype("datetime64[ns]"))
    return out


def label_within_h_trading_days(df: pd.DataFrame, event_dict: dict, h: int, label_name: str):
    """
    对每行 (permno, t)，判断 (t, t_end] 内是否有事件日期
    t_end 用该股票未来第 h 个交易日的日期（shift(-h)）
    """
    x = df.copy().sort_values([PERMNO_COL, DATE_COL])
    x["t_end"] = x.groupby(PERMNO_COL)[DATE_COL].shift(-h)
    x = x.dropna(subset=["t_end"])

    t0 = x[DATE_COL].values.astype("datetime64[ns]")
    t1 = x["t_end"].values.astype("datetime64[ns]")
    perm = x[PERMNO_COL].values.astype(int)

    y = np.zeros(len(x), dtype=np.int8)
    for p in np.unique(perm):
        idx = np.where(perm == p)[0]
        arr = event_dict.get(int(p))
        if arr is None or len(arr) == 0:
            continue
        left  = np.searchsorted(arr, t0[idx], side="right")   # > t
        right = np.searchsorted(arr, t1[idx], side="right")   # <= t_end
        y[idx] = (right > left).astype(np.int8)

    x[label_name] = y
    return x.drop(columns=["t_end"])

def main():
    with open(CONFIG_DIR / "config.yaml", "r", encoding="utf-8") as f:
        # 使用 safe_load 避免执行任意代码，这是安全实践
        data = yaml.safe_load(f)
    
    H_DIV = data['H_label']

    div_ev = pd.read_parquet(DATA_DIR / "interim" / "stage1" / "div_ev.parquet")
    df_full_feat = pd.read_parquet(DATA_DIR / "interim" / "stage2" / "df_full_feat.parquet")

    div_dict = build_event_dict(div_ev, "DCLRDT")
    df_full_labeled = label_within_h_trading_days(df_full_feat, div_dict, h=H_DIV, label_name="y_div_10d")

    output_dir = Path("./data/interim/stage3")
    output_dir.mkdir(parents=True, exist_ok=True)

    df_full_labeled.to_parquet(output_dir / 'df_full_labeled.parquet', index=False)

    print(f"文件已成功保存至: {output_dir}")

if __name__ == "__main__":
    main()