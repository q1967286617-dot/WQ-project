from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from . import CONFIG_DIR, DATA_DIR, PROJECT_ROOT

PERMNO_COL = "PERMNO"
DATE_COL = "DlyCalDt"


def build_fixed_universe(table_b_path: Path, start_all: str, end_all: str, chunksize: int = 500_000):
    permno_set = set()
    reader = pd.read_csv(table_b_path, usecols=[PERMNO_COL, DATE_COL], chunksize=chunksize)
    for chunk in reader:
        chunk[DATE_COL] = pd.to_datetime(chunk[DATE_COL], errors="coerce")
        chunk = chunk.dropna(subset=[PERMNO_COL, DATE_COL])
        mask = (chunk[DATE_COL] >= start_all) & (chunk[DATE_COL] <= end_all)
        permno_set.update(chunk.loc[mask, PERMNO_COL].astype(int).unique())
    return permno_set


def load_market_data_full(table_b_path: Path, permno_set: set, start_all: str, end_all: str, chunksize: int = 500_000):
    usecols = [
        PERMNO_COL, "Ticker", DATE_COL, "SICCD",
        # 核心价格与收益
        "DlyClose", "DlyPrc", "DlyVol", "DlyRet", "ShrOut",
        # OHLC（用于日内振幅、ATR、开盘跳空特征）
        "DlyHigh", "DlyLow", "DlyOpen",
        # 流动性（用于 Bid-Ask Spread、成交笔数特征）
        "DlyBid", "DlyAsk", "DlyNumTrd",
        # 复权因子（备用）
        "DlyCumFacPr", "DlyCumFacShr",
    ]
    out = []
    reader = pd.read_csv(table_b_path, usecols=usecols, chunksize=chunksize, low_memory=True)

    for chunk in reader:
        chunk[DATE_COL] = pd.to_datetime(chunk[DATE_COL], errors="coerce")
        chunk = chunk.dropna(subset=[PERMNO_COL, DATE_COL])
        chunk[PERMNO_COL] = chunk[PERMNO_COL].astype(int)

        mask = (
            chunk[PERMNO_COL].isin(permno_set)
            & (chunk[DATE_COL] >= start_all)
            & (chunk[DATE_COL] <= end_all)
        )
        out.append(chunk.loc[mask])

    df = pd.concat(out, ignore_index=True)
    df = df.drop_duplicates(subset=[PERMNO_COL, DATE_COL])
    df = df.sort_values([PERMNO_COL, DATE_COL]).reset_index(drop=True)
    return df


def load_div_events(
    table_a_path: Path,
    div_distcd: list,
    permno_col: str = PERMNO_COL,
    event_date_col: str = "DCLRDT",
) -> pd.DataFrame:
    ev = pd.read_csv(table_a_path, usecols=[
        permno_col, event_date_col, "DISTCD",
        "DIVAMT",   # 每股分红金额（用于金额特征）
        "EXDT",     # 除息日（用于时间结构特征）
        "FACSHR",   # 股本复权因子（备用）
    ])
    ev[permno_col] = ev[permno_col].astype(int)
    ev[event_date_col] = pd.to_datetime(ev[event_date_col], errors="coerce")
    ev["EXDT"] = pd.to_datetime(ev["EXDT"], errors="coerce")
    ev["DIVAMT"] = pd.to_numeric(ev["DIVAMT"], errors="coerce")
    ev["FACSHR"] = pd.to_numeric(ev["FACSHR"], errors="coerce")

    ev = ev.dropna(subset=[permno_col, event_date_col]).drop_duplicates()
    ev = ev.sort_values([permno_col, event_date_col]).reset_index(drop=True)
    ev = ev[ev["DISTCD"].isin(div_distcd)]
    return ev.drop(columns=["DISTCD"])


def standardize_events(
    events_df: pd.DataFrame,
    permno_col: str = PERMNO_COL,
    event_date_col: str = "DCLRDT",
) -> pd.DataFrame:
    ev = events_df.rename(columns={permno_col: "permno", event_date_col: "event_date"}).copy()
    ev["permno"] = ev["permno"].astype(int)
    ev["event_date"] = pd.to_datetime(ev["event_date"], errors="coerce")
    ev = ev.dropna(subset=["permno", "event_date"]).drop_duplicates()
    ev = ev.sort_values(["permno", "event_date"]).reset_index(drop=True)
    return ev


def main():
    with open(CONFIG_DIR / "config.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    table_a_path = PROJECT_ROOT / data["tableA_path"]
    table_b_path = PROJECT_ROOT / data["tableB_path"]

    start_all = data["start_all"]
    end_all = data["end_all"]
    div_distcd = data["div_distcd"]

    permno_set = build_fixed_universe(table_b_path, start_all, end_all)
    df_full_raw = load_market_data_full(table_b_path, permno_set, start_all, end_all)
    div_ev = load_div_events(table_a_path, div_distcd)

    output_dir = DATA_DIR / "interim" / "stage1"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_full_raw.to_parquet(output_dir / "df_full_raw.parquet", index=False)
    div_ev.to_parquet(output_dir / "div_ev.parquet", index=False)

    print(f"Data Loaded: {output_dir}")


if __name__ == "__main__":
    main()