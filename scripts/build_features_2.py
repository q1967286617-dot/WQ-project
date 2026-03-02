from __future__ import annotations

from typing import Optional

from pathlib import Path

import numpy as np
import pandas as pd

from load import PERMNO_COL, DATE_COL


def add_calendar_features(df: pd.DataFrame):
    """对应你 notebook 的 derive_time_features（保留核心部分即可）"""
    df = df.copy()
    dt = df[DATE_COL].dt

    df["weekday"] = dt.weekday.astype("int8")
    df["month"]   = dt.month.astype("int8")
    df["quarter"] = dt.quarter.astype("int8")
    df["weekofyear"] = dt.isocalendar().week.astype("int16")
    df["dayofyear"]  = dt.dayofyear.astype("int16")

    # 周期编码（示例：weekday / month / dayofyear）
    df["weekday_sin"] = np.sin(2*np.pi*df["weekday"]/7)
    df["weekday_cos"] = np.cos(2*np.pi*df["weekday"]/7)
    df["month_sin"]   = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"]   = np.cos(2*np.pi*df["month"]/12)
    df["doy_sin"]     = np.sin(2*np.pi*df["dayofyear"]/366)
    df["doy_cos"]     = np.cos(2*np.pi*df["dayofyear"]/366)

    df["is_month_start"] = dt.is_month_start.astype("int8")
    df["is_month_end"]   = dt.is_month_end.astype("int8")
    return df


def build_div_event_features(div_ev: pd.DataFrame, recent_n=3):
    """
    在事件表粒度构造“分红节奏”特征，并保证严格因果：
    - gap 由 (当前事件 - 上一次事件)
    - 所有统计量都 shift(1)：表示“在当前事件发生之前”你能知道的历史统计
    """
    ev = div_ev.copy()
    ev["prev"] = ev.groupby(PERMNO_COL)["DCLRDT"].shift(1)
    ev["gap"]  = (ev["DCLRDT"] - ev["prev"]).dt.days

    g = ev.groupby(PERMNO_COL, group_keys=False)

    # 历史事件计数（强烈建议加：后面缺失值填充时模型能区分“真缺失” vs “有历史”）
    ev["div_count_exp"] = g["DCLRDT"].cumcount()  # 0,1,2,...（当前事件的序号）
    ev["div_count_exp"] = ev["div_count_exp"].astype("int16")

    # expanding 统计（shift(1) 防止把“当前 gap”泄露到当前事件之前）
    ev["gap_mean_exp"] = g["gap"].apply(lambda s: s.expanding().mean().shift(1))
    ev["gap_med_exp"]  = g["gap"].apply(lambda s: s.expanding().median().shift(1))
    ev["gap_std_exp"]  = g["gap"].apply(lambda s: s.expanding().std().shift(1))

    # 最近 N 次事件 rolling
    N = int(recent_n)
    ev["gap_mean_rN"] = g["gap"].apply(lambda s: s.rolling(N, min_periods=2).mean().shift(1))
    ev["gap_std_rN"]  = g["gap"].apply(lambda s: s.rolling(N, min_periods=2).std().shift(1))

    cols = [PERMNO_COL, "DCLRDT", "div_count_exp",
            "gap_mean_exp", "gap_med_exp", "gap_std_exp",
            "gap_mean_rN", "gap_std_rN"]
    return ev[cols].sort_values([PERMNO_COL, "DCLRDT"]).reset_index(drop=True)

def build_causal_features_full(df_full: pd.DataFrame, div_ev: pd.DataFrame, div_event_feats: pd.DataFrame):
    df = df_full.copy()

    # 0) 强制 dtype + 排序（rolling/merge_asof 的生命线）
    df[PERMNO_COL] = df[PERMNO_COL].astype(int)
    df[DATE_COL]   = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, PERMNO_COL]).reset_index(drop=True)

    # 1) Calendar features
    df = add_calendar_features(df)

    # 2) 分红：最近一次分红日期（严格 backward）
    # merge_asof 要求：left/right 按 (by, on) 排序
    div_ev2 = div_ev[[PERMNO_COL, "DCLRDT"]].copy()
    div_ev2 = div_ev2.sort_values(["DCLRDT", PERMNO_COL]).reset_index(drop=True)
    
    df = pd.merge_asof(
        df, div_ev2,
        left_on=DATE_COL, right_on="DCLRDT",
        by=PERMNO_COL,
        direction="backward",
        allow_exact_matches=True,
    )
    # 断言：绝不允许未来事件对齐
    bad = df["DCLRDT"].notna() & (df["DCLRDT"] > df[DATE_COL])
    assert int(bad.sum()) == 0
    df = df.rename(columns={"DCLRDT": "last_div_dclrdt"})
    df["days_since_last_div"] = (df[DATE_COL] - df["last_div_dclrdt"]).dt.days

    # 3) 分红节奏统计（事件侧特征 asof 回日频）
    div_event_feats2 = div_event_feats.sort_values(["DCLRDT", PERMNO_COL]).reset_index(drop=True)
    div_event_feats2 = div_event_feats2.rename(columns={"DCLRDT": "div_feat_dclrdt"})

    df = pd.merge_asof(
        df, div_event_feats2,
        left_on=DATE_COL, right_on="div_feat_dclrdt",
        by=PERMNO_COL,
        direction="backward",
        allow_exact_matches=True,
    )
    df = df.drop(columns=["div_feat_dclrdt"])  # 关键：不要让它留在最终特征里

    eps = 1e-6
    df["gap_cv_exp"] = df["gap_std_exp"] / (df["gap_mean_exp"] + eps)

    df["time_to_med_exp"] = df["days_since_last_div"] - df["gap_med_exp"]
    df["z_to_med_exp"]    = df["time_to_med_exp"] / (df["gap_std_exp"] + eps)

    df["time_to_mean_rN"] = df["days_since_last_div"] - df["gap_mean_rN"]
    df["z_to_mean_rN"]    = df["time_to_mean_rN"] / (df["gap_std_rN"] + eps)

    # 4) 价格/成交/收益 rolling（全量算，val/test 会自然继承训练期历史）
    g = df.groupby(PERMNO_COL, sort=False)

    df["log_mkt_cap"] = np.log(df["DlyPrc"].abs() * df["ShrOut"] + 1)

    df["ret_5d"] = g["DlyRet"].transform(lambda s: s.rolling(5).sum())
    df["ret_21d"] = g["DlyRet"].transform(lambda s: s.rolling(21).sum())
    df["vol_5d"] = g["DlyRet"].transform(lambda s: s.rolling(5).std())
    df["vol_21d"] = g["DlyRet"].transform(lambda s: s.rolling(21).std())
    df["turnover_5d"] = (df["DlyVol"] / df["ShrOut"]).groupby(df[PERMNO_COL]).transform(lambda s: s.rolling(5).mean())
    df["turnover_21d"] = (df["DlyVol"] / df["ShrOut"]).groupby(df[PERMNO_COL]).transform(lambda s: s.rolling(21).mean())
    df["price_to_high"] = df["DlyPrc"] / g["DlyPrc"].transform(lambda s: s.rolling(5).max())
    df["volume_spike"] = df["DlyVol"] / g["DlyVol"].transform(lambda s: s.rolling(21).mean())
    df["vol_ratio"] = df["vol_5d"] / (df["vol_21d"] + eps)
    df["turnover_ratio"] = df["turnover_5d"] / (df["turnover_21d"] + eps)

    # 5) 行业横截面特征（当日 SIC 两位行业的平均收益）
    df["industry"] = df["SICCD"].fillna(0).astype(str).str[:2]
    df["ind_avg_ret"] = df.groupby(["industry", DATE_COL])["DlyRet"].transform("mean")
    df["ret_rel_to_ind"] = df["DlyRet"] - df["ind_avg_ret"]

    df["has_div_history"] = df["div_count_exp"].notna().astype("int8")
    df["div_count_exp"] = df["div_count_exp"].fillna(0).astype("int16")
    df["ind_avg_ret"] = df["ind_avg_ret"].fillna(0.0)

    return df


def main():
    div_ev = pd.read_parquet('./data/interim/stage1/div_ev.parquet')
    df_full_raw = pd.read_parquet('./data/interim/stage1/df_full_raw.parquet')
    
    div_event_feats = build_div_event_features(div_ev, recent_n=3)
    df_full_feat = build_causal_features_full(df_full_raw, div_ev, div_event_feats)

    output_dir = Path("./data/interim/stage2")
    output_dir.mkdir(parents=True, exist_ok=True)

    div_event_feats.to_parquet(output_dir / 'div_event_feats.parquet', index=False)
    df_full_feat.to_parquet(output_dir / 'df_full_feat.parquet', index=False)

    print(f"文件已成功保存至: {output_dir}")

if __name__ == "__main__":
    main()