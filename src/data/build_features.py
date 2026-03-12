from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from . import DATA_DIR
from .load import PERMNO_COL, DATE_COL


def add_calendar_features(df: pd.DataFrame):
    """日历特征：周期性编码 + 月首月末标记"""
    df = df.copy()
    dt = df[DATE_COL].dt

    df["weekday"]    = dt.weekday.astype("int8")
    df["month"]      = dt.month.astype("int8")
    df["quarter"]    = dt.quarter.astype("int8")
    df["weekofyear"] = dt.isocalendar().week.astype("int16")
    df["dayofyear"]  = dt.dayofyear.astype("int16")

    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"]     = np.sin(2 * np.pi * df["dayofyear"] / 366)
    df["doy_cos"]     = np.cos(2 * np.pi * df["dayofyear"] / 366)

    df["is_month_start"] = dt.is_month_start.astype("int8")
    df["is_month_end"]   = dt.is_month_end.astype("int8")
    return df


def build_div_event_features(div_ev: pd.DataFrame, recent_n: int = 3) -> pd.DataFrame:
    """
    在事件表粒度构造分红节奏特征 + DIVAMT 金额特征 + EXDT 时间结构特征。
    所有统计量均 shift(1)，确保不泄露当前事件信息。
    """
    ev = div_ev.copy()

    # ── 分红间隔（gap） ────────────────────────────────────────────────
    ev["prev_dclrdt"] = ev.groupby(PERMNO_COL)["DCLRDT"].shift(1)
    ev["gap"] = (ev["DCLRDT"] - ev["prev_dclrdt"]).dt.days

    g = ev.groupby(PERMNO_COL, group_keys=False)

    # 历史事件计数（0,1,2,...）
    ev["div_count_exp"] = g["DCLRDT"].cumcount().astype("int16")

    # 间隔 expanding/rolling 统计（shift(1) 防止泄露当前事件）
    ev["gap_mean_exp"] = g["gap"].apply(lambda s: s.expanding().mean().shift(1))
    ev["gap_med_exp"]  = g["gap"].apply(lambda s: s.expanding().median().shift(1))
    ev["gap_std_exp"]  = g["gap"].apply(lambda s: s.expanding().std().shift(1))

    N = int(recent_n)
    ev["gap_mean_rN"] = g["gap"].apply(lambda s: s.rolling(N, min_periods=2).mean().shift(1))
    ev["gap_std_rN"]  = g["gap"].apply(lambda s: s.rolling(N, min_periods=2).std().shift(1))

    # ── DIVAMT 金额特征 ───────────────────────────────────────────────
    # div_amt_last：上一次分红金额
    ev["div_amt_last"] = g["DIVAMT"].shift(1)

    # div_amt_mean_exp：历史分红金额 expanding 均值
    ev["div_amt_mean_exp"] = g["DIVAMT"].apply(lambda s: s.expanding().mean().shift(1))

    # div_amt_chg：当前金额相对上次的变动（使用 shift(1) 的上次金额，因果安全）
    ev["div_amt_chg"] = ev["DIVAMT"] - ev["div_amt_last"]

    # div_amt_dir_mean_r3：近 3 次金额变动方向均值（+1 涨，-1 跌，0 不变），shift(1)
    ev["_amt_sign"] = (
        ev.groupby(PERMNO_COL)["DIVAMT"]
        .apply(lambda s: np.sign(s.diff()))
        .reset_index(level=0, drop=True)
    )
    ev["div_amt_dir_mean_r3"] = (
        ev.groupby(PERMNO_COL)["_amt_sign"]
        .apply(lambda s: s.rolling(3, min_periods=1).mean().shift(1))
        .reset_index(level=0, drop=True)
    )
    ev = ev.drop(columns=["_amt_sign"])

    # div_no_cut_streak：到上一次事件为止，连续未削减分红的次数
    def _no_cut_streak(s: pd.Series) -> pd.Series:
        result = []
        streak = 0
        prev_val = np.nan
        for val in s:
            result.append(float(streak))
            if pd.notna(val) and pd.notna(prev_val):
                streak = streak + 1 if val >= prev_val else 0
            prev_val = val
        return pd.Series(result, index=s.index)

    ev["div_no_cut_streak"] = (
        g["DIVAMT"].apply(_no_cut_streak)
        .reset_index(level=0, drop=True)
    )

    # ── EXDT 时间结构特征 ─────────────────────────────────────────────
    # decl_to_exdt_days：申报日到除息日的天数（先算列，再用 groupby）
    ev["decl_to_exdt_days"] = (ev["EXDT"] - ev["DCLRDT"]).dt.days

    # decl_to_exdt_mean_exp：历史申报到除息天数 expanding 均值（shift(1)）
    ev["decl_to_exdt_mean_exp"] = (
        ev.groupby(PERMNO_COL, group_keys=False)["decl_to_exdt_days"]
        .apply(lambda s: s.expanding().mean().shift(1))
        .reset_index(level=0, drop=True)
    )

    cols = [
        PERMNO_COL, "DCLRDT",
        # 间隔节奏
        "div_count_exp",
        "gap_mean_exp", "gap_med_exp", "gap_std_exp",
        "gap_mean_rN", "gap_std_rN",
        # DIVAMT 金额
        "div_amt_last", "div_amt_mean_exp", "div_amt_chg",
        "div_amt_dir_mean_r3", "div_no_cut_streak",
        # EXDT 时间结构
        "decl_to_exdt_days", "decl_to_exdt_mean_exp",
    ]
    return ev[cols].sort_values([PERMNO_COL, "DCLRDT"]).reset_index(drop=True)


def build_causal_features_full(
    df_full: pd.DataFrame,
    div_ev: pd.DataFrame,
    div_event_feats: pd.DataFrame,
) -> pd.DataFrame:
    df = df_full.copy()
    eps = 1e-6

    # ── 0) 强制 dtype + 排序 ──────────────────────────────────────────
    df[PERMNO_COL] = df[PERMNO_COL].astype(int)
    df[DATE_COL]   = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, PERMNO_COL]).reset_index(drop=True)

    # ── 1) 日历特征 ───────────────────────────────────────────────────
    df = add_calendar_features(df)

    # ── 2) 最近一次分红日期（严格 backward asof） ─────────────────────
    div_ev2 = div_ev[[PERMNO_COL, "DCLRDT"]].copy()
    # merge_asof 要求右侧 on 键全局有序，只按日期排序
    div_ev2 = div_ev2.sort_values("DCLRDT").reset_index(drop=True)

    df = pd.merge_asof(
        df, div_ev2,
        left_on=DATE_COL, right_on="DCLRDT",
        by=PERMNO_COL,
        direction="backward",
        allow_exact_matches=True,
    )
    bad = df["DCLRDT"].notna() & (df["DCLRDT"] > df[DATE_COL])
    assert int(bad.sum()) == 0, "merge_asof leakage detected"
    df = df.rename(columns={"DCLRDT": "last_div_dclrdt"})
    df["days_since_last_div"] = (df[DATE_COL] - df["last_div_dclrdt"]).dt.days

    # ── 3) 分红节奏统计（事件侧特征 asof 回日频） ─────────────────────
    div_event_feats2 = (
        div_event_feats
        .sort_values("DCLRDT")   # merge_asof 要求右侧 on 键全局有序
        .reset_index(drop=True)
        .rename(columns={"DCLRDT": "div_feat_dclrdt"})
    )

    df = pd.merge_asof(
        df, div_event_feats2,
        left_on=DATE_COL, right_on="div_feat_dclrdt",
        by=PERMNO_COL,
        direction="backward",
        allow_exact_matches=True,
    )
    df = df.drop(columns=["div_feat_dclrdt"])

    # 间隔派生特征
    df["gap_cv_exp"]      = df["gap_std_exp"] / (df["gap_mean_exp"] + eps)
    df["time_to_med_exp"] = df["days_since_last_div"] - df["gap_med_exp"]
    df["z_to_med_exp"]    = df["time_to_med_exp"] / (df["gap_std_exp"] + eps)
    df["time_to_mean_rN"] = df["days_since_last_div"] - df["gap_mean_rN"]
    df["z_to_mean_rN"]    = df["time_to_mean_rN"] / (df["gap_std_rN"] + eps)

    # ── 4) 价格/成交/收益 rolling 特征 ───────────────────────────────
    g = df.groupby(PERMNO_COL, sort=False)

    df["log_mkt_cap"]  = np.log(df["DlyPrc"].abs() * df["ShrOut"] + 1)
    df["ret_5d"]       = g["DlyRet"].transform(lambda s: s.rolling(5).sum())
    df["ret_21d"]      = g["DlyRet"].transform(lambda s: s.rolling(21).sum())
    df["vol_5d"]       = g["DlyRet"].transform(lambda s: s.rolling(5).std())
    df["vol_21d"]      = g["DlyRet"].transform(lambda s: s.rolling(21).std())

    turnover_raw = df["DlyVol"] / (df["ShrOut"] + eps)
    df["turnover_5d"]  = turnover_raw.groupby(df[PERMNO_COL]).transform(lambda s: s.rolling(5).mean())
    df["turnover_21d"] = turnover_raw.groupby(df[PERMNO_COL]).transform(lambda s: s.rolling(21).mean())

    # price_to_high：修正为 252 日窗口（年内最高价比值）
    df["price_to_high"] = df["DlyPrc"] / g["DlyPrc"].transform(lambda s: s.rolling(252, min_periods=20).max())
    df["volume_spike"]  = df["DlyVol"] / g["DlyVol"].transform(lambda s: s.rolling(21).mean())
    df["vol_ratio"]     = df["vol_5d"] / (df["vol_21d"] + eps)
    df["turnover_ratio"] = df["turnover_5d"] / (df["turnover_21d"] + eps)

    # ── 5) OHLC 衍生特征 ─────────────────────────────────────────────
    high  = pd.to_numeric(df["DlyHigh"],  errors="coerce")
    low   = pd.to_numeric(df["DlyLow"],   errors="coerce")
    open_ = pd.to_numeric(df["DlyOpen"],  errors="coerce")
    close = pd.to_numeric(df["DlyClose"], errors="coerce")

    # daily_range_pct：当日振幅 (H-L)/L
    df["daily_range_pct"] = (high - low) / (low.abs() + eps)

    # True Range：max(H-L, |H-prev_close|, |L-prev_close|)
    prev_close = close.groupby(df[PERMNO_COL]).shift(1)
    tr = pd.concat([
        (high - low).rename("hl"),
        (high - prev_close).abs().rename("hpc"),
        (low  - prev_close).abs().rename("lpc"),
    ], axis=1).max(axis=1)
    df["atr_5d"]   = tr.groupby(df[PERMNO_COL]).transform(lambda s: s.rolling(5,  min_periods=1).mean())
    df["atr_21d"]  = tr.groupby(df[PERMNO_COL]).transform(lambda s: s.rolling(21, min_periods=1).mean())
    df["atr_ratio"] = df["atr_5d"] / (df["atr_21d"] + eps)

    # open_gap_pct：开盘价相对昨日收盘的跳空幅度
    df["open_gap_pct"] = (open_ - prev_close) / (prev_close.abs() + eps)

    # ── 6) 流动性特征 ─────────────────────────────────────────────────
    bid = pd.to_numeric(df["DlyBid"], errors="coerce")
    ask = pd.to_numeric(df["DlyAsk"], errors="coerce")
    mid = (bid + ask) / 2.0

    # bid_ask_spread：当日相对价差（half-spread / mid），无效报价填 NaN
    raw_spread = (ask - bid) / (2.0 * mid.abs() + eps)
    raw_spread = raw_spread.where((bid > 0) & (ask > 0) & (ask >= bid), np.nan)
    df["bid_ask_spread"]    = raw_spread
    df["bid_ask_spread_5d"] = raw_spread.groupby(df[PERMNO_COL]).transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )

    num_trd = pd.to_numeric(df["DlyNumTrd"], errors="coerce")
    df["log_num_trd"]   = np.log1p(num_trd.fillna(0))
    df["num_trd_spike"] = num_trd / (
        num_trd.groupby(df[PERMNO_COL]).transform(lambda s: s.rolling(21, min_periods=1).mean()) + eps
    )

    # ── 7) 行业横截面特征 ─────────────────────────────────────────────
    df["industry"]       = df["SICCD"].fillna(0).astype(str).str[:2]
    df["ind_avg_ret"]    = df.groupby(["industry", DATE_COL])["DlyRet"].transform("mean")
    df["ret_rel_to_ind"] = df["DlyRet"] - df["ind_avg_ret"]

    # ── 8) 缺失值填充 ─────────────────────────────────────────────────
    df["has_div_history"] = df["div_count_exp"].notna().astype("int8")
    df["div_count_exp"]   = df["div_count_exp"].fillna(0).astype("int16")
    df["ind_avg_ret"]     = df["ind_avg_ret"].fillna(0.0)

    # DIVAMT / EXDT 特征（无分红历史时填 0）
    for col in [
        "div_amt_last", "div_amt_mean_exp", "div_amt_chg",
        "div_amt_dir_mean_r3", "div_no_cut_streak",
        "decl_to_exdt_days", "decl_to_exdt_mean_exp",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # OHLC / 流动性特征（数据缺失时填 0）
    for col in [
        "daily_range_pct", "atr_5d", "atr_21d", "atr_ratio", "open_gap_pct",
        "bid_ask_spread", "bid_ask_spread_5d", "log_num_trd", "num_trd_spike",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    return df


def main():
    stage1_dir = DATA_DIR / "interim" / "stage1"
    div_ev = pd.read_parquet(stage1_dir / "div_ev.parquet")
    df_full_raw = pd.read_parquet(stage1_dir / "df_full_raw.parquet")

    div_event_feats = build_div_event_features(div_ev, recent_n=3)
    df_full_feat = build_causal_features_full(df_full_raw, div_ev, div_event_feats)

    output_dir = DATA_DIR / "interim" / "stage2"
    output_dir.mkdir(parents=True, exist_ok=True)

    div_event_feats.to_parquet(output_dir / "div_event_feats.parquet", index=False)
    df_full_feat.to_parquet(output_dir / "df_full_feat.parquet", index=False)

    print(f"Features built: {output_dir}")


if __name__ == "__main__":
    main()