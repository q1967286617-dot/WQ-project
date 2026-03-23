from __future__ import annotations

import numpy as np
import pandas as pd


def build_equal_weight_benchmark(
    panel: pd.DataFrame,
    min_price: float,
    ret_col: str = "exec_ret_1d",
    # Fix 1: 新增成本参数，与策略成本口径对齐
    cost_bps_one_way: float = 0.0,
    # Fix 2: 新增持仓期参数，与策略持仓期对齐
    holding_td: int = 1,
    # Fix 3: 新增宇宙过滤参数，与策略可投资宇宙对齐
    turnover_quantile_min: float = 0.0,
    exclude_div_count_le: int = 0,
) -> pd.DataFrame:
    x = panel.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")

    # ── Fix 3: 与策略宇宙对齐的过滤条件 ──────────────────────────────
    # 原代码只有价格和收益率两个过滤
    price_ok = pd.to_numeric(x["DlyPrc"], errors="coerce").abs().ge(float(min_price))
    ret_ok   = pd.to_numeric(x[ret_col],  errors="coerce").notna()

    # 新增：流动性分位数过滤（与策略 build_daily_candidates 一致）
    if turnover_quantile_min > 0.0 and "turnover_5d" in x.columns:
        thr = (
            x.groupby("date")["turnover_5d"]
            .quantile(float(turnover_quantile_min))
            .rename("_bm_thr")
        )
        x = x.merge(thr, on="date", how="left")
        liq_ok = pd.to_numeric(x["turnover_5d"], errors="coerce").ge(
            pd.to_numeric(x["_bm_thr"], errors="coerce")
        )
        x = x.drop(columns=["_bm_thr"])
    else:
        liq_ok = pd.Series(True, index=x.index)

    # 新增：分红历史过滤（与策略 signal_group != "no_history" 一致）
    if exclude_div_count_le > 0 and "div_count_exp" in x.columns:
        div_ok = pd.to_numeric(x["div_count_exp"], errors="coerce").fillna(0).gt(
            int(exclude_div_count_le)
        )
    else:
        div_ok = pd.Series(True, index=x.index)

    x = x[price_ok & ret_ok & liq_ok & div_ok].copy()

    dates = sorted(x["date"].dropna().unique())
    if not dates:
        return pd.DataFrame(
            columns=["date", "benchmark_ret", "benchmark_n", "is_rebal", "benchmark_nav"]
        )

    # 预计算每日 {permno: ret} 映射，避免在主循环内反复过滤全表
    day_ret_map: dict = {}
    day_permno_set: dict = {}
    for dt, g in x.groupby("date"):
        rets = pd.to_numeric(g[ret_col], errors="coerce")
        day_ret_map[dt]    = dict(zip(g["permno"].astype(int), rets))
        day_permno_set[dt] = set(g["permno"].astype(int).tolist())

    # ── Fix 2: 每 holding_td 日再平衡，与策略持仓期对齐 ──────────────
    # 原代码等价于 holding_td=1（每日全量换仓）
    # ── Fix 1: 再平衡日按真实换手率扣除双边成本 ──────────────────────
    # 原代码 benchmark 完全不扣成本
    cost_rate  = float(cost_bps_one_way) / 10_000.0
    holding_td = max(1, int(holding_td))

    current_permnos: set = set()
    rows = []

    for i, dt in enumerate(dates):
        is_rebal = (i % holding_td == 0)

        if is_rebal:
            new_permnos = day_permno_set.get(dt, set())

            if not current_permnos:
                # 首次建仓：只有买入单边成本
                cost = cost_rate
            else:
                # 计算真实换手率：退出旧持仓 + 买入新持仓的平均比例
                n_exit  = len(current_permnos - new_permnos)
                n_enter = len(new_permnos - current_permnos)
                side_turnover = (
                    n_exit  / max(len(current_permnos), 1)
                  + n_enter / max(len(new_permnos), 1)
                ) / 2.0
                # 双边成本 = 卖旧 + 买新，按实际换手比例计
                cost = 2.0 * cost_rate * float(min(side_turnover, 1.0))

            current_permnos = new_permnos
        else:
            cost = 0.0

        # 当日持仓收益：只统计 current_permnos 中仍有行情数据的股票
        day_map   = day_ret_map.get(dt, {})
        held_rets = [
            v for k, v in day_map.items()
            if k in current_permnos and not pd.isna(v)
        ]
        gross_ret = float(np.mean(held_rets)) if held_rets else 0.0

        rows.append({
            "date":          dt,
            "benchmark_ret": gross_ret - cost,
            "benchmark_n":   len(current_permnos),
            "is_rebal":      int(is_rebal),
        })

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out["benchmark_ret"] = out["benchmark_ret"].fillna(0.0)
    out["benchmark_nav"] = (1.0 + out["benchmark_ret"]).cumprod()
    return out
